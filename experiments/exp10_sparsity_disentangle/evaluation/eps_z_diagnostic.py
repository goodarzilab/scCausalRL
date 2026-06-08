"""ε-vs-z entanglement / KL-health diagnostic for a shift Causal-VAE checkpoint.

Question this answers (user concern): does the latent z actually carry information,
or is z ≈ ε (the SCM is a no-op and the exogenous noise carries everything)? And
did the KL regularize ε toward N(0,I), or is ε an unregularized dump of the cell
state?

For a sample of control + perturbed cells we encode ε = encoder(x) and propagate
z = (I − A^T)^{-1} ε (the deterministic eval DAG), then report, side by side for
BOTH representations:

  • silhouette(perturbed vs control)       — how cleanly each space separates the
  • 5-fold logreg AUROC(perturbed/control)   two populations (info content)
  • mean per-dim Wasserstein-1(pert‖ctrl)  — distributional separation, scale-aware
  • mean |off-diagonal correlation|         — disentanglement (lower = better)

Plus z-vs-ε divergence (is the SCM doing anything?):
  • ‖z − ε‖₂ / ‖ε‖₂  (per cell, mean)       — the "did A move the representation"
                                              number; ≈ the A-bypass ratio but on
                                              REAL encodings (≈0 ⇒ z = ε ⇒ dead SCM)
  • mean per-dim W1(z marginal ‖ ε marginal) and mean |corr(z_i, ε_i)|

Plus KL health on ε (did the prior squeeze it?):
  • mean μ², mean σ², mean per-dim KL to N(0,1), and # active units (KL>0.01)

Usage:
  python evaluation/eps_z_diagnostic.py --checkpoint <ckpt> --config <yaml> \
      --out results/grid/<cell>/eps_z --tag <label> [--max_cells 40000 --n_each 4000]
"""
import os, sys, argparse, json, warnings
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings('ignore')

from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from models.causal_vae import CausalVAE


def build_model(ckpt, device):
    sd = ckpt['model_state_dict']
    cfg_m = (ckpt.get('config', {}) or {})['model']
    gene_names = list(ckpt['gene_names']); n_genes = len(gene_names)
    nb = int(sd['encoder.batch_emb.weight'].shape[0]) if 'encoder.batch_emb.weight' in sd else 0
    nm = int(sd['encoder.modality_emb.weight'].shape[0]) if 'encoder.modality_emb.weight' in sd else 0
    model = CausalVAE(
        n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
        tau_init=cfg_m.get('tau_min', cfg_m.get('tau_init', 1.0)),
        n_dag_samples=cfg_m.get('n_dag_samples', 16), dagma_s=cfg_m.get('dagma_s', 1.0),
        n_modalities=nm, n_batches=nb,
        decoder_type=cfg_m.get('decoder_type', 'linear'),
        decoder_hidden=int(cfg_m.get('decoder_hidden', 1024)),
        decoder_layers=int(cfg_m.get('decoder_layers', 2)),
        use_shift_encoder=bool(cfg_m.get('use_shift_encoder', True)),
        shift_label_dim=int(cfg_m.get('shift_label_dim', 50)),
        shift_hidden=int(cfg_m.get('shift_hidden', 1024)),
        shift_n_layers=int(cfg_m.get('shift_n_layers', 1)),
        shift_output_activation=cfg_m.get('shift_output_activation', 'none'),
    ).to(device)
    model.load_state_dict(sd); model.eval()
    return model, gene_names


def mean_marginal_w1(A, B):
    """Mean over dims of 1-D Wasserstein-1 between matched columns of A and B."""
    return float(np.mean([wasserstein_distance(A[:, i], B[:, i]) for i in range(A.shape[1])]))


def offdiag_corr(M):
    c = np.corrcoef(M.T)
    d = c.shape[0]
    return float(np.abs(c[~np.eye(d, dtype=bool)]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default='results/eps_z')
    ap.add_argument('--tag', default='model')
    ap.add_argument('--max_cells', type=int, default=40000)
    ap.add_argument('--n_each', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    import yaml, scipy.sparse
    config = yaml.safe_load(open(args.config))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, gene_names = build_model(ckpt, device)

    from data.load_replogle import load_replogle
    dd = load_replogle(
        path=config['data']['replogle_path'], n_hvg=config['data']['n_hvg'],
        cell_line=config['data'].get('replogle_cell_line', 'k562'),
        max_cells=min(args.max_cells, config['data'].get('replogle_max_cells', args.max_cells)),
        train_frac=config['data'].get('single_pert_train_frac', 0.8),
        val_frac=config['data'].get('single_pert_val_frac', 0.1),
        split_seed=config['data'].get('split_seed', 42))
    ad = dd['adata']
    X = ad.X.toarray() if scipy.sparse.issparse(ad.X) else np.asarray(ad.X)
    X = X.astype('float32')
    assert list(dd['gene_names']) == gene_names, "data gene axis != checkpoint"

    rng = np.random.default_rng(args.seed)
    ctrl_idx = np.array(dd['control_idx'])
    pert_idx = np.array(sorted(set(range(ad.n_obs)) - set(ctrl_idx.tolist())))
    n_each = int(min(args.n_each, len(ctrl_idx), len(pert_idx)))
    sel_c = rng.choice(ctrl_idx, n_each, replace=False)
    sel_p = rng.choice(pert_idx, n_each, replace=False)
    sel = np.concatenate([sel_c, sel_p]); labels = np.array([0] * n_each + [1] * n_each)

    with torch.no_grad():
        Xt = torch.tensor(X[sel], dtype=torch.float32, device=device)
        mu, log_var, eps = model.encoder(Xt)
        A = model.dag.sample_A(n_samples=1).squeeze(0)          # eval = hard gates
        z = model.dag.scm_forward(eps, A)
    eps_np, z_np = eps.cpu().numpy(), z.cpu().numpy()
    mu_np, var_np = mu.cpu().numpy(), np.exp(log_var.cpu().numpy())
    d = eps_np.shape[1]

    # ── per-representation separation / disentanglement ──
    rows = []
    for name, M in [('epsilon (exogenous)', eps_np), ('z (post-SCM latent)', z_np)]:
        sil = float(silhouette_score(M, labels))
        auc = float(cross_val_score(LogisticRegression(max_iter=600), M, labels,
                                    cv=5, scoring='roc_auc').mean())
        w1 = mean_marginal_w1(M[labels == 1], M[labels == 0])     # pert vs ctrl
        rows.append(dict(representation=name, silhouette_pert_ctrl=sil,
                         logreg_auroc_pert_ctrl=auc, w1_pert_vs_ctrl=w1,
                         offdiag_corr=offdiag_corr(M)))
    table = pd.DataFrame(rows)

    # ── z-vs-ε divergence (is the SCM a no-op?) ──
    rel = float(np.mean(np.linalg.norm(z_np - eps_np, axis=1) /
                        (np.linalg.norm(eps_np, axis=1) + 1e-8)))
    w1_zeps = mean_marginal_w1(z_np, eps_np)
    diag_corr = float(np.mean([np.corrcoef(z_np[:, i], eps_np[:, i])[0, 1] for i in range(d)]))

    # ── KL health on ε ──
    kl_per_dim = 0.5 * (mu_np ** 2 + var_np - np.log(var_np + 1e-8) - 1.0)  # [N, d]
    kl_dim_mean = kl_per_dim.mean(0)                                        # [d]
    summary = dict(
        tag=args.tag, checkpoint=args.checkpoint, epoch=ckpt.get('epoch'),
        n_each=n_each, latent_dim=d,
        z_vs_eps_rel_l2=rel, z_vs_eps_w1=w1_zeps, z_vs_eps_diag_corr=diag_corr,
        eps_mean_mu2=float((mu_np ** 2).mean()), eps_mean_var=float(var_np.mean()),
        eps_mean_kl_per_dim=float(kl_dim_mean.mean()),
        eps_active_units=int((kl_dim_mean > 0.01).sum()),
    )

    print(f"\n{'='*78}\nε-vs-z DIAGNOSTIC — {args.tag}  (epoch {ckpt.get('epoch')}, {n_each}+{n_each} cells)\n{'='*78}")
    print(table.to_string(index=False))
    print(f"\n  z-vs-ε  ‖z−ε‖/‖ε‖ = {rel:.4f}   (≈0 ⇒ z=ε ⇒ SCM dead;  larger ⇒ A reshapes z)")
    print(f"  z-vs-ε  mean per-dim W1(z‖ε) = {w1_zeps:.4f}   mean diag corr(z_i,ε_i) = {diag_corr:.3f}")
    print(f"  KL health: mean μ²={summary['eps_mean_mu2']:.3f}  mean σ²={summary['eps_mean_var']:.3f}  "
          f"mean KL/dim={summary['eps_mean_kl_per_dim']:.3f}  active units (KL>0.01)={summary['eps_active_units']}/{d}")

    table.to_csv(os.path.join(args.out, f'eps_z_table_{args.tag}.csv'), index=False)
    json.dump(summary, open(os.path.join(args.out, f'eps_z_summary_{args.tag}.json'), 'w'), indent=2)
    print(f"\nsaved → {args.out}/eps_z_table_{args.tag}.csv  +  eps_z_summary_{args.tag}.json")


if __name__ == '__main__':
    main()
