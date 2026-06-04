import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.causal_vae import CausalVAE
from utils.bio_priors import load_collectri, get_tf_target_pairs, compute_grn_auprc


def eval_dag(config, vae_path, data_dict=None, device=None, output_dir='logs/dag_eval'):
    """
    Evaluate the learned DAG against known biology (CollecTRI).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating DAG quality on {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if data_dict is None:
        from data.load_norman import load_norman
        data_dict = load_norman(
            path=config['data']['norman_path'],
            n_hvg=config['data']['n_hvg']
        )
    
    gene_names = data_dict['gene_names']
    n_genes = len(gene_names)
    
    # Load model
    ckpt = torch.load(vae_path, map_location=device)
    cfg_model = config['model']
    model = CausalVAE(
        n_genes=n_genes,
        latent_dim=cfg_model['latent_dim'],
        tau_init=cfg_model['tau_min'],
        n_dag_samples=cfg_model['n_dag_samples'],
        dagma_s=cfg_model['dagma_s'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    results = {}
    
    # ============================================================
    # 1. Get DAG statistics
    # ============================================================
    with torch.no_grad():
        edge_probs = model.dag.get_edge_probs().cpu().numpy()  # [d, d]
        A_hard = (edge_probs > 0.5).astype(float)
        n_edges = int(A_hard.sum())
        avg_in_degree = A_hard.sum(axis=0).mean()
        avg_out_degree = A_hard.sum(axis=1).mean()
        
        # DAGMA constraint
        A_tensor = torch.tensor(A_hard, dtype=torch.float32, device=device)
        h_val = model.dag.dagma_constraint(A_tensor).item()
    
    results['n_edges'] = n_edges
    results['avg_in_degree'] = float(avg_in_degree)
    results['avg_out_degree'] = float(avg_out_degree)
    results['dagma_h'] = float(h_val)
    
    print(f"\nDAG Statistics:")
    print(f"  Number of edges: {n_edges}")
    print(f"  Average in-degree: {avg_in_degree:.2f}")
    print(f"  Average out-degree: {avg_out_degree:.2f}")
    print(f"  DAGMA acyclicity h(A): {h_val:.6f} (0 = perfect DAG)")
    
    # ============================================================
    # 2. Project to gene space: G_feat = D @ A @ D^T
    # ============================================================
    with torch.no_grad():
        D = model.decoder.D.detach()  # [n_genes, d]
        G_feat = model.decoder.project_dag_to_genes(A_tensor).cpu().numpy()  # [n_genes, n_genes]
    
    G_feat_abs = np.abs(G_feat)
    
    # ============================================================
    # 3. Compare against CollecTRI
    # ============================================================
    print("\nLoading CollecTRI reference network...")
    collectri_net = load_collectri(organism='human')
    
    if len(collectri_net) > 0:
        ct_pairs = get_tf_target_pairs(collectri_net, gene_names)
        print(f"  CollecTRI pairs in our gene set: {len(ct_pairs)}")
        
        if len(ct_pairs) > 0:
            # AUPRC
            auprc = compute_grn_auprc(G_feat_abs, ct_pairs, gene_names, n_genes)
            results['auprc'] = float(auprc)
            print(f"\n  AUPRC vs CollecTRI: {auprc:.4f}")
            
            # Top-K precision
            n_total = n_genes * n_genes
            G_flat = G_feat_abs.ravel()
            top_k_vals = [100, 500, 1000]
            for k in top_k_vals:
                if k > len(G_flat):
                    continue
                top_k_indices = np.argsort(G_flat)[::-1][:k]
                true_set = set(tf * n_genes + tgt for tf, tgt in ct_pairs)
                hits = sum(1 for idx in top_k_indices if idx in true_set)
                precision_k = hits / k
                results[f'precision_at_{k}'] = float(precision_k)
                print(f"  Precision@{k}: {precision_k:.4f}")
    else:
        print("  CollecTRI not available, skipping GRN evaluation")
        results['auprc'] = None
    
    # ============================================================
    # 4. Top genes per latent program
    # ============================================================
    D_np = model.decoder.D.detach().cpu().numpy()  # [n_genes, d]
    n_top = 5
    latent_dim = cfg_model['latent_dim']
    top_genes_per_program = {}
    
    print(f"\nTop {n_top} genes per latent program (showing first 10 programs):")
    for prog_idx in range(min(10, latent_dim)):
        loadings = np.abs(D_np[:, prog_idx])
        top_gene_indices = np.argsort(loadings)[::-1][:n_top]
        top_genes = [gene_names[i] for i in top_gene_indices]
        top_genes_per_program[prog_idx] = top_genes
        print(f"  Program {prog_idx:3d}: {', '.join(top_genes)}")
    
    results['top_genes_per_program'] = top_genes_per_program
    
    # ============================================================
    # 5. Visualizations
    # ============================================================
    print("\nGenerating visualizations...")
    
    # Edge probability heatmap
    try:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        ax = axes[0]
        sns.heatmap(edge_probs, ax=ax, cmap='Blues', 
                    xticklabels=False, yticklabels=False, vmin=0, vmax=1)
        ax.set_title(f'DAG Edge Probabilities\n({n_edges} edges with p>0.5)', fontsize=14)
        ax.set_xlabel('Target node')
        ax.set_ylabel('Source node')
        
        # G_feat heatmap (subset to top genes for readability)
        ax = axes[1]
        n_show = min(50, n_genes)
        # Find top-varying genes
        gene_var = G_feat_abs.sum(axis=1) + G_feat_abs.sum(axis=0)
        top_gene_idx = np.argsort(gene_var)[::-1][:n_show]
        G_sub = G_feat[np.ix_(top_gene_idx, top_gene_idx)]
        gene_labels = [gene_names[i] for i in top_gene_idx]
        
        sns.heatmap(G_sub, ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=gene_labels if n_show <= 30 else False,
                    yticklabels=gene_labels if n_show <= 30 else False)
        ax.set_title(f'Gene-space GRN (top {n_show} genes)\nG_feat = D @ A @ D^T', fontsize=14)
        if n_show <= 30:
            ax.tick_params(axis='x', rotation=90, labelsize=6)
            ax.tick_params(axis='y', rotation=0, labelsize=6)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'dag_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {save_path}")
        results['visualization_path'] = save_path
    except Exception as e:
        print(f"  Visualization failed: {e}")
    
    print("\n=== DAG Evaluation Complete ===")
    return results
