"""Preprocess the Joung TF Atlas (GSE217460 singles + GSE217066 combinatorial) into
compact, gene-axis-aligned h5ads for world-model training + combo/triple eval.

Why a preprocessing step: the raw TFAtlas h5ad is stored DENSE (1.15M x 37,528 ≈ 172 GB),
so it cannot be loaded whole. We chunk-read it, restrict to a shared gene axis, keep the
relevant cells, and write a small sparse h5ad.

Shared gene axis = (combinatorial 8,506 genes)  ∪  (TF-perturbation genes present in the
atlas). This guarantees (a) every held-out combo/triple gene is on the axis (clean eval
alignment) and (b) every perturbed TF gene has a row in the control-PCA pert_label so the
shift encoder can embed it.

Controls = GFP (TFORF3549) + mCherry (TFORF3550) overexpression cells (negative controls,
present in the processed atlas). TF identity per cell = the gene symbol after the ORF id
(`TFORF0867-NR1H2` → NR1H2); isoforms of the same gene are merged at the gene level.

Outputs (same axis):
  joung_singles_proc.h5ad   obs: TF_gene, is_control            (training)
  joung_combo_proc.h5ad     obs: TF_set, set_size               (combo/triple eval)
"""
import os
import argparse
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd

ATLAS = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/GSE217460_210322_TFAtlas_raw.h5ad"
COMBO = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/GSE217066_210715_combinatorial.h5ad"
CTRL_NAMES = {'GFP', 'mCherry'}


def gene_of(tf_str):
    """`TFORF0867-NR1H2` -> 'NR1H2' (gene symbols may contain '-', so split once)."""
    return tf_str.split('-', 1)[1] if '-' in tf_str else tf_str


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--ctrl_cells', type=int, default=30000, help="subsample control cells")
    ap.add_argument('--max_per_tf', type=int, default=200, help="cap cells per TF (0=all)")
    ap.add_argument('--chunk', type=int, default=20000)
    ap.add_argument('--max_chunks', type=int, default=0, help="smoke: stop after N chunks (0=all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- axis from combinatorial genes + atlas TF genes ----
    print("[1] reading combinatorial var + atlas obs/var (backed)...", flush=True)
    cb = ad.read_h5ad(COMBO, backed='r')
    combo_genes = list(map(str, cb.var_names))
    a = ad.read_h5ad(ATLAS, backed='r')
    var = list(map(str, a.var_names)); var_idx = {g: i for i, g in enumerate(var)}
    tf = a.obs['TF'].astype(str).values
    gof = np.array([gene_of(t) for t in tf])
    is_ctrl = np.isin(gof, list(CTRL_NAMES))
    tf_genes_present = sorted(set(gof[~is_ctrl]) & set(var))
    axis = sorted(set(combo_genes) | set(tf_genes_present))
    axis = [g for g in axis if g in var_idx]
    axis_cols = np.array([var_idx[g] for g in axis])
    print(f"    atlas cells={a.n_obs}  TF genes present={len(tf_genes_present)}  "
          f"combo genes={len(combo_genes)}  → axis={len(axis)}", flush=True)

    # ---- choose atlas cells: real-TF (gene in axis) capped per TF + control subsample ----
    axis_set = set(axis)
    real = (~is_ctrl) & np.isin(gof, list(axis_set))
    rng = np.random.default_rng(0)
    keep_mask = np.zeros(a.n_obs, dtype=bool)
    if args.max_per_tf and args.max_per_tf > 0:
        df = pd.DataFrame({'g': gof, 'i': np.arange(a.n_obs)})[real]
        for g, sub in df.groupby('g'):
            idx = sub['i'].values
            keep_mask[idx if len(idx) <= args.max_per_tf else rng.choice(idx, args.max_per_tf, replace=False)] = True
    else:
        keep_mask[real] = True
    ctrl_all = np.where(is_ctrl)[0]
    if args.ctrl_cells and 0 < args.ctrl_cells < len(ctrl_all):
        ctrl_keep = rng.choice(ctrl_all, args.ctrl_cells, replace=False)
    else:
        ctrl_keep = ctrl_all                                  # ctrl_cells<=0 → ALL control
    keep_mask[ctrl_keep] = True
    keep = np.where(keep_mask)[0]
    print(f"    keeping {len(keep)} atlas cells (real-TF {int((real&keep_mask).sum())} + "
          f"control {len(ctrl_keep)})", flush=True)

    # ---- chunked read of atlas X[keep][:, axis_cols] → sparse (direct h5py: X is a
    #      DENSE contiguous dataset, so row-slices are efficient contiguous reads) ----
    print("[2] chunk-reading atlas X via h5py (dense 172GB → axis-subset sparse)...", flush=True)
    import h5py
    n = a.n_obs
    blocks, kept_rows = [], []
    n_chunks_done = 0
    with h5py.File(ATLAS, 'r') as hf:
        Xds = hf['X']
        for s in range(0, n, args.chunk):
            if args.max_chunks and n_chunks_done >= args.max_chunks:
                break
            n_chunks_done += 1
            e = min(s + args.chunk, n)
            local = keep[(keep >= s) & (keep < e)] - s          # rows to keep in this chunk
            if local.size == 0:
                continue
            block = np.asarray(Xds[s:e])                         # [chunk, 37528] dense
            sub = block[local][:, axis_cols].astype(np.float32)
            blocks.append(sp.csr_matrix(sub))
            kept_rows.extend((s + local).tolist())
            if (s // args.chunk) % 5 == 0:
                print(f"    ...{e}/{n}  kept so far={len(kept_rows)}", flush=True)
    X = sp.vstack(blocks).tocsr()
    kept_rows = np.array(kept_rows)
    obs = pd.DataFrame({'TF_gene': gof[kept_rows], 'is_control': is_ctrl[kept_rows]},
                       index=[f"atlas_{i}" for i in kept_rows])
    out = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=axis))
    sp_out = os.path.join(args.out_dir, 'joung_singles_proc.h5ad')
    out.write_h5ad(sp_out)
    print(f"[3] wrote {sp_out}  shape={out.shape}  "
          f"#TF genes={obs.loc[~obs.is_control,'TF_gene'].nunique()}  ctrl={int(obs.is_control.sum())}", flush=True)

    # ---- combinatorial: raw counts aligned to the SAME axis ----
    print("[4] processing combinatorial (raw counts → axis)...", flush=True)
    cbm = ad.read_h5ad(COMBO)                         # full (167k x 8506, small)
    raw = cbm.raw.to_adata() if cbm.raw is not None else cbm
    rvar = {g: i for i, g in enumerate(map(str, raw.var_names))}
    Xr = raw.X.toarray() if sp.issparse(raw.X) else np.asarray(raw.X)
    Xc = np.zeros((Xr.shape[0], len(axis)), dtype=np.float32)
    present = [(j, rvar[g]) for j, g in enumerate(axis) if g in rvar]
    for j, ri in present:
        Xc[:, j] = Xr[:, ri]
    tfset = cbm.obs['TF'].astype(str).values
    size = np.array([t.count(',') + 1 for t in tfset])
    cobs = pd.DataFrame({'TF_set': tfset, 'set_size': size}, index=list(cbm.obs_names))
    cout = ad.AnnData(X=sp.csr_matrix(Xc), obs=cobs, var=pd.DataFrame(index=axis))
    cp = os.path.join(args.out_dir, 'joung_combo_proc.h5ad')
    cout.write_h5ad(cp)
    print(f"[5] wrote {cp}  shape={cout.shape}  axis-genes-in-combo-raw={len(present)}/{len(axis)}  "
          f"pairs={int((size==2).sum())} triples={int((size==3).sum())}", flush=True)
    print("DONE")


if __name__ == '__main__':
    main()
