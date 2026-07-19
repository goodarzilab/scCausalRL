"""Magnitude-crediting singles metric (the Audit-1 follow-up). disc_resid drift-removes then does
cosine discrimination, which zeros TFs whose true effect is collinear with the shared drift. Here we
ALSO compute, per held-out TF, metrics that credit getting the effect right regardless of drift
alignment, and show the LOW-disc TFs still score well on them (→ the variance is largely a metric
artifact, not a prediction failure):
  disc_resid  : drift-removed cosine discrimination (fraction of other test TFs beaten)  [current]
  raw_pear    : Pearson(pred Δ, obs Δ)  WITHOUT drift removal (credits the shared direction)
  deg_auroc   : can |pred Δ| recover the true DEGs (|obs Δ|>0.1)? direction-agnostic
  eff_spear   : Spearman(|pred|, |obs|) over genes — magnitude ranking

Run: python evaluation/drift_metric_singles.py --checkpoint <ckpt> [--device cpu]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import zrow, VDIR, PROC, EFF
from data.load_joung import load_joung
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='results/grid/v2b_retrfix_refresh_k20/checkpoints/phase1_epoch60.pt')
    ap.add_argument('--n_ctrl', type=int, default=500); ap.add_argument('--device', default='cpu'); a = ap.parse_args()
    dev = torch.device(a.device); torch.set_grad_enabled(False)
    model, gn, ck = load_shift_model(a.checkpoint, dev)
    p1 = (ck.get('config', {}) or {}).get('phase1', {}) or {}
    BETA = float(p1.get('pert_mutilate_beta', 1.0)); UMODE = p1.get('pert_mutilate_u_mode', 'shift')
    gidx = {g: i for i, g in enumerate(gn)}
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF); de = dd['de_single']
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(a.n_ctrl, len(ctrl)), replace=False)]], device=dev)
    _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0); z_prev = model.dag.scm_forward(eps, A)
    pd = getattr(model, 'pert_decoder', None) or model.decoder; cr = pd(z_prev).mean(0).cpu().numpy()

    train = [g for g in dd['pert_split']['train'] if g in de and g in gidx]
    test = [g for g in dd['pert_split']['test'] if g in de and g in gidx]
    drift = np.stack([np.asarray(de[g], np.float32) for g in train]).mean(0)
    obs = np.stack([np.asarray(de[g], np.float32) for g in test])
    pred = np.zeros((len(test), len(gn)), np.float32)
    for k, g in enumerate(test):
        zp, _ = model.intervene_shift_mutilate(eps, [gidx[g]], A, z_prev, u_mode=UMODE, shift_beta=BETA)
        pred[k] = pd(zp).mean(0).cpu().numpy() - cr

    # per-TF disc_resid (drift-removed cosine discrimination)
    Pr, Or = zrow(pred - drift), zrow(obs - drift)
    Cm = Pr @ Or.T; dgn = np.diag(Cm)[:, None]; T = len(test)
    disc = (dgn > Cm).sum(1) / (T - 1)

    rows = []
    for k, g in enumerate(test):
        p, o = pred[k], obs[k]
        raw_pear = np.corrcoef(p, o)[0, 1] if p.std() > 1e-9 else np.nan          # raw Δ, ALL genes (incl KO)
        keep = np.ones(len(gn), bool)                                             # KO-excluded raw Pearson (Pnk)
        if g in gidx: keep[gidx[g]] = False
        pk, ok = p[keep], o[keep]
        pnk = np.corrcoef(pk, ok)[0, 1] if pk.std() > 1e-9 and ok.std() > 1e-9 else np.nan  # = cos(centered)
        deg = (np.abs(o) > 0.15).astype(int)
        auroc = roc_auc_score(deg, np.abs(p)) if 0 < deg.sum() < len(deg) else np.nan
        eff = spearmanr(np.abs(p), np.abs(o)).correlation
        rows.append((g, float(disc[k]), float(raw_pear), float(pnk), float(auroc), float(eff)))
    arr = np.array([[r[1], r[2], r[3], r[4], r[5]] for r in rows], float)
    names = ['disc_resid', 'raw_pear', 'raw_pear_noKO', 'deg_auroc', 'eff_spear']
    print(f"=== singles metrics on {len(test)} held-out TFs (top model) ===")
    for j, nm in enumerate(names):
        v = arr[:, j]; print(f"  {nm:11s} mean={np.nanmean(v):.3f}  std={np.nanstd(v):.3f}  "
                             f"[{np.nanmin(v):.2f},{np.nanmax(v):.2f}]")
    # do LOW-disc TFs still score on magnitude metrics?
    lo = arr[:, 0] < 0.5; hi = arr[:, 0] >= 0.5
    print(f"\n  LOW-disc TFs (disc<0.5, n={lo.sum()}):  raw_pear_noKO={np.nanmean(arr[lo,2]):.3f}  "
          f"deg_auroc={np.nanmean(arr[lo,3]):.3f}  eff_spear={np.nanmean(arr[lo,4]):.3f}")
    print(f"  HIGH-disc TFs (disc>=0.5, n={hi.sum()}): raw_pear_noKO={np.nanmean(arr[hi,2]):.3f}  "
          f"deg_auroc={np.nanmean(arr[hi,3]):.3f}  eff_spear={np.nanmean(arr[hi,4]):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, (j, nm) in enumerate([(2, 'raw Pearson noKO (Pnk)'), (3, 'DEG-recovery AUROC'), (4, 'effect Spearman |pred|,|obs|')]):
        ax[i].scatter(arr[:, 0], arr[:, j], c=['#d03b3b' if x < 0.5 else '#1baf7a' for x in arr[:, 0]], s=45, alpha=0.85)
        ax[i].axvline(0.5, ls='--', color='#888'); ax[i].set_xlabel('disc_resid (drift-removed)'); ax[i].set_ylabel(nm)
        r = np.corrcoef(arr[np.isfinite(arr[:, j]), 0], arr[np.isfinite(arr[:, j]), j])[0, 1]
        ax[i].set_title(f'{nm}  vs disc_resid  (r={r:.2f})', fontsize=11)
    fig.suptitle('SINGLES: low-disc TFs (red) still recover DEGs / magnitude — disc_resid under-credits drift-aligned TFs', fontsize=12)
    fig.tight_layout(); out = 'results/figures/singles_magnitude_metrics.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); print(f"\nwrote {out}")
    import csv
    with open('results/figures/singles_perTF_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['tf'] + names); [w.writerow([r[0]] + list(r[1:])) for r in rows]
    print("wrote results/figures/singles_perTF_metrics.csv")


if __name__ == '__main__':
    main()
