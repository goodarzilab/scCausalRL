"""Stage 2 (state_env): score a world model's predicted deltas with the SAME cell_eval
MetricsEvaluator STATE was scored with, on STATE's real held-out K562 data.

Construction (matches STATE's basal+shift form): for each real perturbed cell of pert p,
predicted expression = (a sampled real non-targeting control cell) + delta_model[p],
clipped at 0. Control cells predict themselves. So pred_delta ≈ delta_model and real_delta
is STATE's real held-out delta → pearson_delta / DE / discrimination are directly
comparable to STATE's cached k562_results.csv.

Run (state_env):
  python score_celleval.py --delta results/exp16_delta.npz --label exp16 \
    --out results/exp16_celleval
"""
import os, argparse, json
import numpy as np
import anndata as ad
import pandas as pd
from cell_eval import MetricsEvaluator

SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--arcinstitute--ST-SE-Replogle/snapshots/e324967ff4cea5ec199e29bcbb5c1f00e5b9d69c")
REAL_DEFAULT = f"{SNAP}/fewshot/k562/eval_best.ckpt/adata_real.h5ad"
HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = 'non-targeting'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta', required=True, help='npz from predict_deltas.py')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--real', default=REAL_DEFAULT, help='STATE adata_real.h5ad (fewshot or zeroshot)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip_slow', action='store_true',
                    help='skip pearson_edistance + clustering_agreement (not in the reported table)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    npz = np.load(args.delta, allow_pickle=True)
    perts = list(map(str, npz['perts'])); delta = npz['delta'].astype(np.float32)
    dhvg = list(map(str, npz['state_hvg']))
    dmap = {p: i for i, p in enumerate(perts)}
    print(f"[{args.label}] delta: {delta.shape}, {len(perts)} perts", flush=True)

    real = ad.read_h5ad(args.real)
    state_hvg = json.load(open(f"{HERE}/results/state_hvg.json"))
    # adata_real var index i ↔ state_hvg[i]; align delta columns to that order
    col = {g: j for j, g in enumerate(dhvg)}
    delta_full = np.zeros((len(perts), len(state_hvg)), np.float32)
    keep_g = [i for i, g in enumerate(state_hvg) if g in col]
    delta_full[:, keep_g] = delta[:, [col[state_hvg[i]] for i in keep_g]]

    g = real.obs['gene'].astype(str).values
    keep = np.isin(g, perts) | (g == CTRL)
    real = real[keep].copy()
    g = real.obs['gene'].astype(str).values
    Xr = real.X.toarray() if hasattr(real.X, 'toarray') else np.asarray(real.X)
    Xr = Xr.astype(np.float32)
    ctrl_rows = np.where(g == CTRL)[0]
    print(f"[{args.label}] scoring cells: {real.n_obs} ({len(ctrl_rows)} control)", flush=True)

    # build predictions: control→self; pert cell→ sampled control + delta
    Xp = Xr.copy()
    pmask = g != CTRL
    samp = ctrl_rows[rng.integers(0, len(ctrl_rows), size=int(pmask.sum()))]
    di = np.array([dmap[p] for p in g[pmask]])
    Xp[pmask] = np.clip(Xr[samp] + delta_full[di], 0, None)

    pred = ad.AnnData(X=Xp, obs=real.obs.copy(), var=real.var.copy())
    ev = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert=CTRL,
                          pert_col='gene', de_method='wilcoxon', outdir=args.out)
    skip = ['pearson_edistance', 'clustering_agreement'] if args.skip_slow else None
    res, agg = ev.compute(profile='full', skip_metrics=skip, break_on_error=False)
    res.write_csv(f"{args.out}/{args.label}_results.csv")
    agg.write_csv(f"{args.out}/{args.label}_agg.csv")
    rdf = res.to_pandas()
    key = [c for c in ['pearson_delta', 'de_direction_match', 'de_spearman_lfc_sig',
                       'discrimination_score_cosine', 'mae_delta'] if c in rdf.columns]
    print(f"\n[{args.label}] n={len(rdf)} | " + "  ".join(f"{k}={rdf[k].mean():.4f}" for k in key))
    print(f"saved → {args.out}/")


if __name__ == '__main__':
    main()
