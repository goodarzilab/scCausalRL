"""Extract observed Briggs COMBO Δ (2-TF pairs) in the WM 4366-gene space, for the cov+ head fine-tune.

Mirrors evaluation/briggs/reconstruct_targets.py: per-cell TF presence from obsm['TF_UMIs_filt'] +
uns['TF_names'] (collapse 800 isoforms -> gene symbols); for each 2-TF gene-set with >=min_cells cells
CONTAINING it, Δ = mean(containing screen cells) - hESC-background, on genes SHARED with the WM (matched
CP10k-lognorm), placed into the WM 4366-gene axis (zero-fill non-shared). Background = Briggs hESC start
(pluripotent top-frac of screen cells) — the natural control for hESC-derived combos, analogous to Joung
'combo - control'. Writes npz {obs_delta[n,G_wm], pairs[n,2], n_contain[n]} for briggs_head_finetune.py.
"""
import os, argparse
import numpy as np, scipy.sparse as sp, anndata as ad
from collections import Counter

VOCAB = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
BRIGGS = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"
PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']
CTRL = {'ctrl', 'control', 'hesc', 'unperturbed', 'nt'}


def lognorm(X):
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5ad', default=BRIGGS); ap.add_argument('--sample_col', default='sample')
    ap.add_argument('--min_cells', type=int, default=10); ap.add_argument('--plurip_pct', type=float, default=0.10)
    ap.add_argument('--kmax', type=int, default=2, help='extract sets up to this size (2 = pairs)')
    ap.add_argument('--out', default='/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing/results/briggs/briggs_combo_delta.npz')
    a = ap.parse_args()

    # WM gene axis = Joung proc var_names (asserted == frozen WM gene axis)
    js = ad.read_h5ad(VOCAB + "/joung_singles_proc_v2b.h5ad", backed='r')
    wm_genes = list(map(str, js.var_names)); wm_idx = {g: i for i, g in enumerate(wm_genes)}
    G = len(wm_genes)

    A = ad.read_h5ad(a.h5ad)
    genes = list(map(str, A.var_names)); samp = A.obs[a.sample_col].astype(str).values
    screen = ~np.array([s.lower() in CTRL for s in samp])
    shared = [g for g in genes if g in wm_idx]
    print(f"[briggs] {A.shape[0]} cells | shared with WM: {len(shared)}/{G} genes | screen cells {int(screen.sum())}", flush=True)

    bln_s = lognorm(A[:, shared].X)                                     # [N, |shared|] matched CP10k-lognorm
    # hESC-start background (pluripotent top-frac of screen cells)
    pmk = [shared.index(g) for g in PLURI if g in shared]
    pscore = bln_s[:, pmk].mean(1)
    thr = np.quantile(pscore[screen], 1 - a.plurip_pct)
    hesc = screen & (pscore >= thr)
    bulk = bln_s[hesc].mean(0)
    print(f"[bg] hESC-start background = {int(hesc.sum())} pluripotent screen cells (top {a.plurip_pct:.0%})", flush=True)

    # per-cell TF gene presence (collapse isoforms)
    TF = A.obsm['TF_UMIs_filt']; TF = (TF.toarray() if sp.issparse(TF) else np.asarray(TF)) > 0
    iso_gene = [str(t).split('|')[0] for t in A.uns['TF_names']]
    ug = sorted(set(iso_gene)); g2c = {g: i for i, g in enumerate(ug)}
    cg = np.zeros((A.shape[0], len(ug)), bool)
    for j, g in enumerate(iso_gene):
        cg[:, g2c[g]] |= TF[:, j]

    # unique exact TF gene-sets among screen cells (size 2..kmax), restricted to WM-gene TFs
    exact = Counter()
    for i in np.where(screen)[0]:
        gs = tuple(sorted(g for g in (ug[j] for j in np.where(cg[i])[0]) if g in wm_idx))
        if 2 <= len(gs) <= a.kmax:
            exact[gs] += 1

    obs_delta, pairs, ncontain = [], [], []
    shared_wm_pos = np.array([wm_idx[g] for g in shared])               # where each shared gene lands in WM axis
    for gs, _ in exact.items():
        cols = [g2c[g] for g in gs]
        contain = cg[:, cols].all(1) & screen
        n = int(contain.sum())
        if n < a.min_cells:
            continue
        d_shared = bln_s[contain].mean(0) - bulk                        # [|shared|]
        d = np.zeros(G, np.float32); d[shared_wm_pos] = d_shared        # -> WM 4366-gene axis
        obs_delta.append(d); pairs.append(list(gs)); ncontain.append(n)
    obs_delta = np.stack(obs_delta) if obs_delta else np.zeros((0, G), np.float32)
    print(f"[extract] {len(pairs)} Briggs pairs with >= {a.min_cells} containing cells "
          f"(of {len(exact)} unique screen sets, size 2..{a.kmax})", flush=True)
    # summary: how many involve canonical HSPC TFs
    hspc = {'TAL1', 'LMO2', 'ETV2', 'GATA2', 'RUNX1', 'SPI1', 'GFI1B', 'HHEX', 'MEIS1', 'FLI1', 'ERG'}
    n_hspc = sum(len(set(p) & hspc) >= 1 for p in pairs)
    print(f"[extract] {n_hspc}/{len(pairs)} pairs involve >=1 canonical HSPC TF", flush=True)
    np.savez_compressed(a.out, obs_delta=obs_delta, pairs=np.array(pairs, dtype=object),
                        n_contain=np.array(ncontain), shared_idx=shared_wm_pos)   # shared_idx: valid WM genes (mask non-shared)
    print(f"saved -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
