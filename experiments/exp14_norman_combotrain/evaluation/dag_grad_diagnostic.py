"""Gradient-flow diagnostic: is the perturbation loss training the DAG, or does
the learned shift S^p bypass it?

Loads the trained checkpoint, builds a real cell-sentence batch, and measures:
  1. per-module grad norms from the perturbation loss (encoder/decoder/dag.theta/
     dag.weights/shift_encoder/shift_scale)
  2. grad to dag.weights & dag.theta from the PERT loss alone vs from the
     spectral+sparsity regularizers (which shrink A)
  3. A's actual contribution to the prediction: ‖z_post − (ε+S)‖ / ‖ε+S‖
  4. ‖S^p‖ vs ‖ε‖ (does the shift dominate the latent?)
  5. latent z off-diagonal |corr| on CONTROL-only vs perturbed+control mix
     (isolates encoder entanglement from perturbation-induced correlation)

Usage: python evaluation/dag_grad_diagnostic.py [--ckpt ...] [--max_cells 20000]
"""
import os, sys, argparse, glob, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')
import numpy as np, torch, yaml, scipy.sparse
from torch.utils.data import DataLoader
from models.causal_vae import CausalVAE
from training.losses import (perturbation_set_loss_shift, sparsity_loss,
                             spectral_stability_loss)
from data.load_replogle import load_replogle
from data.cell_sentence_dataset import PairedCellSentenceDataset

EXP_DIR = os.path.join(os.path.dirname(__file__), '..')


def module_grad_norms(model):
    def gn(params):
        t = 0.0
        for p in params:
            if p.grad is not None:
                t += float(p.grad.detach().norm(2)) ** 2
        return t ** 0.5
    return {
        'encoder': gn(model.encoder.parameters()),
        'decoder': gn(model.decoder.parameters()),
        'dag.theta': (float(model.dag.theta.grad.norm(2)) if model.dag.theta.grad is not None else 0.0),
        'dag.weights': (float(model.dag.weights.grad.norm(2)) if model.dag.weights.grad is not None else 0.0),
        'shift_encoder': gn(model.shift_encoder.parameters()),
        'shift_scale': (float(model.shift_scale.grad.norm(2)) if model.shift_scale.grad is not None else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(EXP_DIR, 'results/phase1/checkpoints/phase1_epoch150.pt'))
    ap.add_argument('--config', default=None)
    ap.add_argument('--max_cells', type=int, default=20000)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = ckpt['model_state_dict']; cfg_m = ckpt['config']['model']; p1 = ckpt['config']['phase1']
    gene_names = list(ckpt['gene_names']); n_genes = len(gene_names)
    nb = sd['encoder.batch_emb.weight'].shape[0] if 'encoder.batch_emb.weight' in sd else 0
    nm = sd['encoder.modality_emb.weight'].shape[0] if 'encoder.modality_emb.weight' in sd else 0
    model = CausalVAE(n_genes=n_genes, latent_dim=cfg_m['latent_dim'],
                      tau_init=cfg_m.get('tau_min', 1.0), n_dag_samples=cfg_m.get('n_dag_samples', 16),
                      dagma_s=cfg_m.get('dagma_s', 1.0), n_modalities=nm, n_batches=nb,
                      decoder_type=cfg_m.get('decoder_type', 'linear'),
                      use_shift_encoder=True, shift_label_dim=int(cfg_m.get('shift_label_dim', 50)),
                      shift_hidden=int(cfg_m.get('shift_hidden', 1024)),
                      shift_n_layers=int(cfg_m.get('shift_n_layers', 1))).to(dev)
    model.load_state_dict(sd); model.train()

    cfg = ckpt['config']
    dd = load_replogle(path=cfg['data']['replogle_path'], n_hvg=cfg['data']['n_hvg'],
                       cell_line=cfg['data'].get('replogle_cell_line', 'k562'),
                       max_cells=min(args.max_cells, cfg['data'].get('replogle_max_cells', args.max_cells)),
                       train_frac=cfg['data'].get('single_pert_train_frac', 0.8),
                       val_frac=cfg['data'].get('single_pert_val_frac', 0.1),
                       split_seed=cfg['data'].get('split_seed', 42))
    adata = dd['adata']; pert_split = dd.get('pert_split')
    train_genes = pert_split['train'] if pert_split else None
    sds = PairedCellSentenceDataset(adata=adata, single_perts=dd['single_perts'],
                                    control_idx=dd['control_idx'], gene_names=gene_names,
                                    S=32, n_sentences_per_block=4, min_block_cells=3,
                                    gene_filter=train_genes, seed=42, modality_id_lookup=None)
    loader = DataLoader(sds, batch_size=min(4, len(sds)), shuffle=True, drop_last=True)
    sent = next(iter(loader))
    xc = sent['x_ctrl'].to(dev); xp = sent['x_pert'].to(dev)
    gi = sent['gene_idx'].to(dev); bid = sent['batch_id'].to(dev)
    B, S, G = xc.shape
    print(f"batch: B={B} sentences, S={S} cells, G={G} genes\n")

    # ── 1. perturbation-loss grad flow ──────────────────────────────────────
    model.zero_grad()
    pert_loss, diag = perturbation_set_loss_shift(
        xc, xp, gene_idx=gi, batch_id=bid, modality_id=None, model=model,
        n_dag_samples=1, mse_weight=float(p1.get('pert_mse_weight', 1.0)),
        mmd_weight=float(p1.get('pert_mmd_weight', 1.0)),
        energy_weight=float(p1.get('pert_energy_weight', 0.0)))
    pert_loss.backward()
    gp = module_grad_norms(model)
    print("=== (1) grad norms from the PERTURBATION loss ===")
    for k, v in gp.items():
        print(f"   {k:16s} {v:.4e}")
    print(f"   pert_loss = {float(pert_loss):.4f}\n")

    # ── 2. regularizer grad to A vs pert grad to A ──────────────────────────
    model.zero_grad()
    A = model.dag.sample_A(n_samples=1).squeeze(0)
    reg = (float(p1.get('lambda_sparse', 10.0)) * sparsity_loss(A, model=model,
              mode=p1.get('lambda_sparse_mode', 'product'))
           + float(p1.get('lambda_spectral', 0.1)) * spectral_stability_loss(A, model.dag))
    reg.backward()
    gr = module_grad_norms(model)
    print("=== (2) grad to A: PERT loss vs REGULARIZERS (spectral+sparsity) ===")
    print(f"   dag.weights:  pert={gp['dag.weights']:.4e}   reg={gr['dag.weights']:.4e}   "
          f"ratio pert/reg = {gp['dag.weights']/(gr['dag.weights']+1e-12):.3f}")
    print(f"   dag.theta  :  pert={gp['dag.theta']:.4e}   reg={gr['dag.theta']:.4e}   "
          f"ratio pert/reg = {gp['dag.theta']/(gr['dag.theta']+1e-12):.3f}\n")

    # ── 3. A's contribution to the prediction (bypass) ──────────────────────
    with torch.no_grad():
        xcf = xc.reshape(B * S, G)
        _, _, eps = model.encoder(xcf, batch_id=bid.repeat_interleave(S))
        Sp = model.compute_shift(gi)                              # [B, d]
        Sp_full = Sp.repeat_interleave(S, dim=0)                  # [B*S, d]
        base = eps + Sp_full
        A_eff = model.dag.get_signed_edges()
        d = A_eff.shape[0]
        LHS = torch.eye(d, device=dev) - A_eff.T
        z_post = torch.linalg.solve(LHS, base.T).T
        bypass = float((z_post - base).norm() / (base.norm() + 1e-12))
        sr = float(torch.max(torch.abs(torch.linalg.eigvals(A_eff))).real)
    print("=== (3) does A change the prediction? ===")
    print(f"   spectral_radius(A) = {sr:.3e}")
    print(f"   ‖z_post − (ε+S)‖ / ‖ε+S‖ = {bypass:.3e}   (≈0 ⇒ DAG bypassed)\n")

    # ── 4. shift vs eps magnitude ───────────────────────────────────────────
    print("=== (4) shift vs exogenous magnitude ===")
    print(f"   mean ‖S^p‖ = {float(Sp.norm(dim=1).mean()):.3f}   mean ‖ε‖ = {float(eps.norm(dim=1).mean()):.3f}   "
          f"ratio ‖S‖/‖ε‖ = {float(Sp.norm(dim=1).mean())/ (float(eps.norm(dim=1).mean())+1e-12):.3f}\n")

    # ── 5. latent correlation: control-only vs perturbed+control ────────────
    with torch.no_grad():
        rng = np.random.default_rng(0)
        cidx = np.array(dd['control_idx'])
        Xall = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.asarray(adata.X)
        Xall = Xall.astype('float32')
        cc = rng.choice(cidx, min(4000, len(cidx)), replace=False)
        pidx = np.array(sorted(set(range(adata.n_obs)) - set(cidx.tolist())))
        pp = rng.choice(pidx, min(4000, len(pidx)), replace=False)
        def zcorr(idx):
            x = torch.tensor(Xall[idx], dtype=torch.float32, device=dev)
            _, _, e = model.encoder(x)
            zz = model.dag.scm_forward(e, model.dag.sample_A(1).squeeze(0)).cpu().numpy()
            c = np.corrcoef(zz.T); off = np.abs(c[~np.eye(c.shape[0], dtype=bool)])
            return float(off.mean())
        zc_ctrl = zcorr(cc); zc_mix = zcorr(np.concatenate([cc, pp]))
    print("=== (5) latent z off-diagonal |corr| (disentanglement) ===")
    print(f"   control-only           = {zc_ctrl:.4f}")
    print(f"   perturbed+control mix  = {zc_mix:.4f}   "
          f"(if mix ≫ control, the 0.47 is perturbation-structure, not encoder entanglement)")
    print("\nNOTE: KL is diagonal-Gaussian on q(ε|x); it does NOT penalize cross-dim "
          "aggregate correlation (needs total-correlation / β-TCVAE for that). There is "
          "NO KL or penalty on the shift S^p.")


if __name__ == '__main__':
    main()
