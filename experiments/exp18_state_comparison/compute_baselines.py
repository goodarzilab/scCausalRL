"""ContextMean + PerturbMean baseline deltas on STATE's 2000-HVG axis (State-paper
definitions), for the K562 held-out test perts. Output as delta .npz so Stage 2
(score_celleval.py) scores them with the identical cell_eval as the models + STATE.

  ContextMean : predict the cell-type mean perturbation profile (same for every pert) →
                delta = mean(all K562 perturbed cells) − mean(K562 controls). Constant.
  PerturbMean : control basal + per-pert offset averaged across the OTHER cell lines
                where the pert was seen (the fewshot signal) →
                delta[p] = mean over {hepg2,jurkat,rpe1} of (mean(p cells) − control mean).
                Perts unseen in all other lines get zero offset (→ predicts control).

Run (base env, from exp16 dir on path):
  python compute_baselines.py --exp_dir ../exp16_multicellline --state_hvg results/state_hvg.json \
    --perts results/perts_state_all.json --out_dir results
"""
import os, sys, json, argparse
import numpy as np, scipy.sparse

LINES = {
    'k562':  '/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad',
    'hepg2': '/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/GSE264667_hepg2_raw_singlecell_01.h5ad',
    'jurkat':'/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/GSE264667_jurkat_raw_singlecell_01.h5ad',
    'rpe1':  '/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/rpe1_normalized_singlecell_01.h5ad',
}


def line_means(path, hvg, perts_keep, cell_line, max_cells=150000):
    from data.load_replogle_celline_aligned import load_replogle_celline_aligned
    dd = load_replogle_celline_aligned(path, target_gene_names=hvg, cell_line=cell_line,
                                       max_cells=max_cells, min_cells_per_pert=5)
    X = dd['adata'].X; X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    ctrl = X[np.array(dd['control_idx'])].mean(0)
    pert_delta = {}
    for g, idx in dd['single_perts'].items():
        if g in perts_keep:
            pert_delta[g] = X[np.array(idx)].mean(0) - ctrl
    allpert = X[[i for i in range(len(X)) if i not in set(dd['control_idx'])]].mean(0) - ctrl
    return ctrl, pert_delta, allpert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True)
    ap.add_argument('--state_hvg', required=True)
    ap.add_argument('--perts', required=True)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(args.exp_dir))
    hvg = json.load(open(args.state_hvg)); perts = json.load(open(args.perts)); pset = set(perts)
    G = len(hvg)

    # ContextMean: from K562 (all perturbed cells mean − control)
    print("[ctx] K562...", flush=True)
    _, _, ctx_delta = line_means(LINES['k562'], hvg, pset, 'k562')

    # PerturbMean: cross-line offset over the other 3 lines
    offsets = {p: [] for p in perts}
    for ln in ['hepg2', 'jurkat', 'rpe1']:
        print(f"[pmean] {ln}...", flush=True)
        _, pdelta, _ = line_means(LINES[ln], hvg, pset, ln)
        for p, dv in pdelta.items():
            offsets[p].append(dv)
    pmean = np.zeros((len(perts), G), np.float32)
    n_seen = 0
    for i, p in enumerate(perts):
        if offsets[p]:
            pmean[i] = np.mean(offsets[p], 0); n_seen += 1
    print(f"[pmean] perts with cross-line signal: {n_seen}/{len(perts)}", flush=True)

    ctx = np.tile(ctx_delta[None], (len(perts), 1)).astype(np.float32)
    np.savez(f"{args.out_dir}/contextmean_delta.npz", perts=np.array(perts), delta=ctx, state_hvg=np.array(hvg))
    np.savez(f"{args.out_dir}/perturbmean_delta.npz", perts=np.array(perts), delta=pmean, state_hvg=np.array(hvg))
    print(f"saved contextmean_delta.npz + perturbmean_delta.npz ({len(perts)} perts)")


if __name__ == '__main__':
    main()
