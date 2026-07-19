"""Joung→Norman transfer: does the Joung TF-atlas world model's compositional do-operator
predict NORMAN combos (a different cell system, K562 vs hESC) for the TFs the two datasets
share?

Setup (restricted, apples-to-apples):
  - Freeze the Joung WM (ugate, gate-aware do-op). It has ESM-based shifts for ANY gene,
    incl. the Norman TFs.
  - Align Norman onto Joung's 9923-gene axis; keep only the ~6735 genes Norman actually
    measures (shared-gene mask) for scoring.
  - Evaluate the combos whose BOTH genes are in Joung's TF vocabulary (~40).
  - For each: do-op (gate) vs additive (Σ Joung single shifts), scored against Norman's
    OBSERVED combo Δ (combo-mean − Norman-control-mean) on the shared DEGs.

Reports raw and drift-removed (subtract the across-combo mean Δ) pearson + discrimination
(is pred_i nearest its own obs_i among the combos). This measures CONTEXT-INVARIANCE of
TF programs, not a like-for-like transfer (hESC↛K562), and is reported as such.

Run (GPU): python transfer_joung_to_norman.py --output rl0_mpc/results/transfer_norman
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from core.wm_loader import load_frozen_wm
from core.wm_operator import WorldModelOperator

EXP19 = '/home/sgoudarzi/scCausalRL/experiments/exp19_nonlinear_compose'
NORMAN_PATH = '/large_storage/goodarzilab/shervin/CausalBioRL/Norman/'


def _pearson(a, b, mask):
    a, b = a[mask], b[mask]
    if a.std() < 1e-8 or b.std() < 1e-8 or len(a) < 3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _discrimination(P, O, mask):
    """Beat-fraction: for each i, is cos(pred_i, obs_i) > cos(pred_i, obs_j) for j≠i,
    on the shared genes. Chance 0.5."""
    P, O = P[:, mask], O[:, mask]
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    On = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-8)
    C = Pn @ On.T                                            # [n, n] cos(pred_i, obs_j)
    n = len(P); wins = 0; tot = 0
    for i in range(n):
        self_c = C[i, i]
        for j in range(n):
            if j == i:
                continue
            wins += (self_c > C[i, j]); tot += 1
    return wins / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default=f'{EXP19}/results/grid/v2_esm_con_ugate/checkpoints/phase1_epoch60.pt')
    ap.add_argument('--norman_path', default=NORMAN_PATH)
    ap.add_argument('--n_ctrl', type=int, default=500)
    ap.add_argument('--min_cells', type=int, default=20)
    ap.add_argument('--output', default='rl0_mpc/results/transfer_norman')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    # ---- frozen Joung WM + gate-aware operator ----
    model, meta = load_frozen_wm(checkpoint_path=args.checkpoint, wm_root=EXP19, device=dev)
    op = WorldModelOperator(model, meta, n_dag_samples=8)
    gene_names = meta['gene_names']
    gidx = meta['gene_to_idx']
    G = len(gene_names)
    A = op.mean_A()

    # ---- Joung control (hESC) ε from the singles atlas (sparse, densify only sample) ----
    import anndata as ad
    from data.load_joung import _lognorm
    S = ad.read_h5ad(f'{EXP19}/data/joung_singles_proc.h5ad')
    Xs = _lognorm(S.X)
    cidx = np.where(S.obs['is_control'].values.astype(bool))[0]
    rng = np.random.default_rng(0)
    sel = cidx[rng.choice(len(cidx), min(args.n_ctrl, len(cidx)), replace=False)]
    xc = torch.tensor(np.asarray(Xs[sel].todense(), np.float32), device=dev)
    _, _, eps_ctrl = model.encoder(xc)
    eps_mean = eps_ctrl.mean(0, keepdim=True)                 # [1, d]
    ctrl_recon = op.decode(op.scm_forward(eps_mean, A)).squeeze(0).cpu().numpy()

    def do_op(gis):
        z, _, _ = op.apply(eps_mean, list(gis), A)            # gate-aware
        return op.decode(z).squeeze(0).cpu().numpy() - ctrl_recon

    def additive(gis):
        tot = np.zeros(G, np.float32)
        for g in gis:
            z, _, _ = op.apply(eps_mean, [g], A)
            tot += op.decode(z).squeeze(0).cpu().numpy() - ctrl_recon
        return tot

    # ---- Norman, aligned to the Joung 9923-gene axis ----
    from data.load_norman_aligned import load_norman_aligned
    import scipy.sparse as sp
    nd = load_norman_aligned(args.norman_path, target_gene_names=gene_names,
                             min_cells_per_pert=args.min_cells)
    Xn = nd['adata'].X
    Xn = (Xn.toarray() if sp.issparse(Xn) else np.asarray(Xn)).astype(np.float32)
    nctrl = np.asarray(nd['control_idx'])
    norman_ctrl_mean = Xn[nctrl].mean(0)
    # shared-gene mask: genes Norman actually measures (nonzero across any cell)
    shared = (np.abs(Xn).sum(0) > 0)
    print(f"[transfer] shared genes (Norman∩Joung axis): {int(shared.sum())}/{G}")

    # combos with BOTH genes in Joung's vocabulary
    rows = []
    for (ga, gb), idxs in nd['combo_perts'].items():
        if ga not in gidx or gb not in gidx:
            continue
        idxs = np.asarray(idxs)
        if len(idxs) < args.min_cells:
            continue
        gis = [gidx[ga], gidx[gb]]
        obs = Xn[idxs].mean(0) - norman_ctrl_mean
        rows.append({'combo': f'{ga}+{gb}', 'gis': gis,
                     'obs': obs, 'do': do_op(gis), 'add': additive(gis)})
    n = len(rows)
    print(f"[transfer] usable Norman combos (both genes Joung TFs, ≥{args.min_cells} cells): {n}")
    if n < 2:
        print("too few combos — abort"); return

    Obs = np.stack([r['obs'] for r in rows])
    Do = np.stack([r['do'] for r in rows]); Add = np.stack([r['add'] for r in rows])
    drift = Obs.mean(0)                                       # Norman's across-combo drift

    def block(P, label):
        thr = 0.05
        pe = np.mean([_pearson(P[i], Obs[i], shared & (np.abs(Obs[i]) > thr)) for i in range(n)])
        # drift-removed
        Pr, Or = P - P.mean(0), Obs - drift
        pe_r = np.mean([_pearson(Pr[i], Or[i], shared & (np.abs(Or[i]) > thr)) for i in range(n)])
        disc = _discrimination(P, Obs, shared)
        disc_r = _discrimination(Pr, Or, shared)
        return {'pearson_delta': round(float(pe), 4), 'pearson_delta_resid': round(float(pe_r), 4),
                'discrimination': round(float(disc), 4), 'discrimination_resid': round(float(disc_r), 4)}

    res = {'n_combos': n, 'shared_genes': int(shared.sum()),
           'do_op': block(Do, 'do'), 'additive': block(Add, 'add')}
    print("\n===== Joung→Norman transfer (context-invariance; hESC WM → K562 combos) =====")
    print(json.dumps(res, indent=2))
    json.dump({**res, 'combos': [r['combo'] for r in rows]},
              open(f'{args.output}/summary.json', 'w'), indent=2)
    print(f"saved → {args.output}/summary.json")


if __name__ == '__main__':
    main()
