"""P4 (v3) — DIRECTION-based observed answer key (fate-axis projection, per user).

Per combo, rho_obs = cosine of the combo's DISPLACEMENT-from-background with the HSPC fate axis
  d = HSPC_pseudobulk - background     (points hESC-screen -> HSPC)
  Delta_obs(C) = mean(cells whose TF-set CONTAINS C) - background
  rho_obs(C)   = <Delta_obs, d> / (||Delta_obs|| ||d||)      (+ unnormalized proj = <Delta,d>/||d||)
This is a MOVEMENT score, not an absolute-position cosine (control+Delta position is dominated by the
hESC baseline and cannot discriminate combos). Two fate axes: Briggs-native (pHSC_mean - screen_bulk)
and Cao HSC deviation signature. Combos scored by CONTAINMENT (Briggs sub-combination augmentation) so
high-MOI cells power each estimate. Directly-observed sets are the target list + P1/P2 anchors.
(Epistasis test stays on directly-observed pairs elsewhere; augmentation used only for this rank target.)

Speed: only the union of the two axes' genes is needed for the displacement, so we subset once.
Writes results/briggs/observed_combo_targets.csv:
  combo(|-joined), k, n_contain, rho_briggs, rho_cao, proj_briggs, proj_cao, is_P1, is_P2
Run: python evaluation/briggs/reconstruct_targets.py --h5ad <all_cells .h5ad> [--min_cells 3]
"""
import os, sys, argparse, numpy as np, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.briggs.hspc_sim import lognorm, build_cao_hsc_signature

CTRL = {'phsc', 'phsc_cfu', 'phep'}
P1 = {'TAL1', 'RPA3', 'ZFP90', 'RNF138', 'LMO2', 'KLF2'}
P2 = {'DDIT3', 'MZF1', 'HOXB3', 'MIER1', 'SIRT6', 'ZNF85'}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']    # hESC-start markers
VOCAB = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h5ad', required=True)
    ap.add_argument('--min_cells', type=int, default=3); ap.add_argument('--sample_col', default='sample')
    ap.add_argument('--background', default='screen_bulk', choices=['screen_bulk', 'hesc', 'hesc_joung', 'joung_ctrl'],
                    help="origin for the fate axis: screen_bulk (mean of all perturbed cells, contaminated by "
                         "failed directions); hesc (clean hESC-start = pluripotent subset of screen cells); "
                         "hesc_joung (pool Briggs-hESC + Joung control, shared genes); joung_ctrl (PURE Joung "
                         "control cells, shared genes) — all cross-dataset modes use MATCHED CP10k-on-shared-genes norm")
    ap.add_argument('--plurip_pct', type=float, default=0.10, help='top-fraction of screen cells by pluripotency score = hESC background')
    ap.add_argument('--out', default='results/briggs/observed_combo_targets.csv'); a = ap.parse_args()
    A = ad.read_h5ad(a.h5ad); genes = list(map(str, A.var_names)); samp = A.obs[a.sample_col].astype(str).values
    Xln = lognorm(A.X); is_ctrl = np.array([s.lower() in CTRL for s in samp]); screen = ~is_ctrl
    phsc_m = np.array([s.lower() == 'phsc' for s in samp])

    # ---- clean hESC-start subset (top pluripotency-score screen cells) ----
    pmk = [genes.index(g) for g in PLURI if g in genes]
    pscore = Xln[:, pmk].mean(1)
    thr = np.quantile(pscore[screen], 1 - a.plurip_pct)
    hesc_mask = screen & (pscore >= thr)
    print(f"[hESC] {int(hesc_mask.sum())} pluripotent screen cells (top {a.plurip_pct:.0%}, score>={thr:.2f}) "
          f"| markers {[g for g in PLURI if g in genes]}", flush=True)

    # ---- Joung control (matched CP10k-on-SHARED-genes normalization) ----
    # Briggs was CP10k-normed over 31915 genes, Joung over 4366 -> different per-cell denominators.
    # Fix: restrict to shared genes FIRST, then lognorm both -> same universe, comparable means.
    need_joung = a.background in ('hesc_joung', 'joung_ctrl')
    if need_joung or a.background == 'hesc':
        import anndata as _ad
        js = _ad.read_h5ad(VOCAB + "/joung_singles_proc_v2b.h5ad")
        jset = set(map(str, js.var_names)); shared = [g for g in genes if g in jset]
        jctrl = js.obs['is_control'].astype(str).str.lower().isin(['true', '1']).values
        Xjc = js[jctrl, shared].X; Xjc = (Xjc.toarray() if sp.issparse(Xjc) else np.asarray(Xjc)).astype(np.float32)
        jctrl_s = lognorm(Xjc).mean(0)                                       # Joung control, shared genes, CP10k lognorm
        sidx_b = [genes.index(g) for g in shared]
        Xb_s = A[:, shared].X; Xb_s = (Xb_s.toarray() if sp.issparse(Xb_s) else np.asarray(Xb_s)).astype(np.float32)
        bln_s = lognorm(Xb_s)                                                # Briggs (all cells), shared genes, matched CP10k
        cc = float(np.corrcoef(jctrl_s, bln_s[hesc_mask].mean(0))[0, 1])
        print(f"[poolability] shared genes={len(shared)} | corr(Joung-control, Briggs-hESC)={cc:+.3f} "
              f"(matched CP10k on shared genes; n_joung_ctrl={int(jctrl.sum())})", flush=True)

    # ---- fate axes (direction) with the chosen background ----
    if need_joung:
        genes = shared; Xln = bln_s                                          # metric on shared genes, matched norm
        bulk = jctrl_s if a.background == 'joung_ctrl' else 0.5 * Xln[hesc_mask].mean(0) + 0.5 * jctrl_s
    elif a.background == 'hesc':
        bulk = Xln[hesc_mask].mean(0)                                        # clean hESC-start origin (Briggs pluripotent)
    else:
        bulk = Xln[screen].mean(0)                                          # legacy screen bulk
    pm = Xln[phsc_m].mean(0)
    bidx = np.where((pm > 0.01) | (bulk > 0.01))[0]             # expressed genes
    d_b = (pm - bulk)[bidx]                                     # Briggs-native fate axis: pHSC - background
    d_c, cidx = build_cao_hsc_signature(genes)                 # Cao HSC deviation signature + its gene idx
    keep = np.array(sorted(set(bidx.tolist()) | set(cidx.tolist())))   # only genes either axis needs
    pos = {g: i for i, g in enumerate(keep)}
    bpos = np.array([pos[g] for g in bidx]); cpos = np.array([pos[g] for g in cidx])
    Xk = Xln[:, keep]; bulk_k = bulk[keep]                     # [N, ~few-thousand]
    print(f"[axes] background={a.background} briggs-axis genes={len(bidx)} cao-axis genes={len(cidx)} "
          f"union={len(keep)} | screen={int(screen.sum())} pHSC={int(phsc_m.sum())}", flush=True)

    # ---- cell x TF-gene presence (collapse 800 isoforms -> 388 genes) ----
    TF = A.obsm['TF_UMIs_filt']; TF = (TF.toarray() if sp.issparse(TF) else np.asarray(TF)) > 0
    iso_gene = [t.split('|')[0] for t in map(str, A.uns['TF_names'])]; ug = sorted(set(iso_gene)); g2c = {g: i for i, g in enumerate(ug)}
    cg = np.zeros((A.shape[0], len(ug)), bool)
    for j, g in enumerate(iso_gene): cg[:, g2c[g]] |= TF[:, j]

    # ---- target combos = unique exact TF gene-sets among screen cells + P1/P2 ----
    from collections import Counter
    exact = Counter()
    for i in np.where(screen)[0]:
        gs = tuple(sorted(ug[j] for j in np.where(cg[i])[0]))
        if 1 <= len(gs) <= 20: exact[gs] += 1
    targets = list(exact.keys())
    for P in (P1, P2):
        t = tuple(sorted(P))
        if t not in exact: targets.append(t)
    print(f"[targets] {len(targets)} directly-observed sets, scoring each by containment", flush=True)

    rows = []
    for gs in targets:
        cols = [g2c[g] for g in gs if g in g2c]
        if len(cols) != len(gs): continue                      # a TF absent from the barcode library
        contain = cg[:, cols].all(1) & screen                  # cells whose TF-set CONTAINS gs
        n = int(contain.sum())
        if n < a.min_cells: continue
        delta = Xk[contain].mean(0) - bulk_k                   # displacement from background (subset genes)
        db, dc = delta[bpos], delta[cpos]
        rows.append(dict(combo='|'.join(gs), k=len(gs), n_contain=n,
                         rho_briggs=cos(db, d_b), rho_cao=cos(dc, d_c),
                         proj_briggs=float(np.dot(db, d_b) / (np.linalg.norm(d_b) + 1e-9)),
                         proj_cao=float(np.dot(dc, d_c) / (np.linalg.norm(d_c) + 1e-9)),
                         is_P1=int(set(gs) == P1), is_P2=int(set(gs) == P2)))
    import csv
    os.makedirs(os.path.dirname(a.out), exist_ok=True); out = a.out
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    rb = np.array([r['rho_briggs'] for r in rows]); rc = np.array([r['rho_cao'] for r in rows]); nc = np.array([r['n_contain'] for r in rows])
    from scipy.stats import spearmanr
    print(f"\n[direction answer key] {len(rows)} combos | n_contain median {int(np.median(nc))} max {int(nc.max())}", flush=True)
    print(f"  rho_briggs: min {rb.min():+.3f} median {np.median(rb):+.3f} max {rb.max():+.3f}", flush=True)
    print(f"  combo-level agreement corr(rho_briggs,rho_cao)={np.corrcoef(rb,rc)[0,1]:+.3f}  spearman={spearmanr(rb,rc).correlation:+.3f}", flush=True)
    for nm, P in (('P1', P1), ('P2', P2)):
        m = [r for r in rows if set(r['combo'].split('|')) == P]
        if m:
            r = m[0]; pct = float((rb < r['rho_briggs']).mean() * 100)
            print(f"  {nm} {sorted(P)}: rho_briggs {r['rho_briggs']:+.3f} (pctile {pct:.0f}), rho_cao {r['rho_cao']:+.3f}, n_contain {r['n_contain']}", flush=True)
    print('wrote', out, flush=True)


if __name__ == '__main__':
    main()
