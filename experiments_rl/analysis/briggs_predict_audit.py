"""Deep-debug: WHY is held-out Briggs combo prediction corr negative? Audit the pipeline end-to-end.

Checks:
 A. Vocab/alignment — test TFs in WM axis (confirmed) + spot-check shared-gene alignment.
 B. OWN-DATASET sanity — does the WM do-op predict JOUNG combos? (upper bound; if low => do-op/pipeline issue,
    not a Briggs problem). Report corr on full-4366 and on the 2800-shared mask.
 C. CROSS-DATASET single-TF — corr(Joung observed single Δ, Briggs single Δ) and corr(WM do-op single Δ,
    Briggs single Δ), per TF, on shared genes, raw + drift-removed. Isolates baseline/direction agreement at k=1.
 D. CROSS-DATASET combo — the -0.10, raw + drift-removed, hESC vs Joung-ctrl background.
 E. Drift dominance — ||drift|| / ||Δ|| for pred and obs (is the signal all shared differentiation program?).
"""
import os, sys, argparse
import numpy as np, scipy.sparse as sp, anndata as ad, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack

V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
BRIGGS = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"
PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']
TFS = ['ETV2', 'TAL1', 'LMO2', 'GATA2', 'RUNX1', 'MEIS1']


def lognorm(X):
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)


def cc(a, b, m):
    a, b = a[m], b[m]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    gene_names = list(map(str, data.gene_names)); g2i = data.gene_to_idx
    A = op.mean_A(); eps0 = data.eps_ctrl_mean.to(dev)
    ctrl_recon = op.decode(data.z0.unsqueeze(0))[0].detach().cpu().numpy()

    @torch.no_grad()
    def wm_delta(genes):
        z = op.apply_sets(eps0, [list(genes)], A)
        return op.decode(z)[0].cpu().numpy() - ctrl_recon                # [G] WM gene axis

    # ---------- B. OWN-DATASET: WM do-op vs JOUNG observed combos ----------
    print("===== B. OWN-DATASET sanity (WM do-op vs JOUNG observed combos) =====", flush=True)
    full = np.arange(len(gene_names))
    jc = []
    for key, de in list(data.de_combo.items()):
        gi = [g2i[t] for t in key if t in g2i]
        if len(gi) < 2:
            continue
        pred = wm_delta(gi); obs = np.asarray(de['lfc'], np.float32)
        jc.append(cc(pred, obs, full))
    print(f"  Joung combos (n={len(jc)}): mean corr(pred, obs) on full 4366 = {np.nanmean(jc):+.3f} "
          f"[{np.nanpercentile(jc,25):+.2f},{np.nanpercentile(jc,75):+.2f}]", flush=True)

    # ---------- Briggs load ----------
    B = ad.read_h5ad(BRIGGS); bgenes = list(map(str, B.var_names)); samp = B.obs['sample'].astype(str).values
    screen = ~np.array([s.lower() in {'ctrl', 'control', 'hesc', 'nt'} for s in samp])
    shared = [g for g in bgenes if g in g2i]
    swm = np.array([g2i[g] for g in shared])                            # WM positions of shared genes
    shared_mask = swm                                                   # corr mask (WM axis)
    bln = lognorm(B[:, shared].X)                                       # [N,|shared|] Briggs, shared, matched norm
    pmk = [shared.index(g) for g in PLURI if g in shared]
    pscore = bln[:, pmk].mean(1); thr = np.quantile(pscore[screen], 0.90)
    hesc = screen & (pscore >= thr)
    bulk_hesc = bln[hesc].mean(0)
    # Joung control bulk on shared (alternative background)
    js = ad.read_h5ad(V + "/joung_singles_proc_v2b.h5ad")
    jctrl = js.obs['is_control'].astype(str).str.lower().isin(['true', '1']).values
    jbulk = lognorm(js[jctrl][:, shared].X).mean(0)
    TF = B.obsm['TF_UMIs_filt']; TF = (TF.toarray() if sp.issparse(TF) else np.asarray(TF)) > 0
    iso = [str(t).split('|')[0] for t in B.uns['TF_names']]; ug = sorted(set(iso)); g2c = {g: i for i, g in enumerate(ug)}
    cg = np.zeros((B.shape[0], len(ug)), bool)
    for j, g in enumerate(iso):
        cg[:, g2c[g]] |= TF[:, j]

    def briggs_delta(tfset, bg):                                        # observed Briggs Δ on WM axis (shared filled)
        cols = [g2c[t] for t in tfset if t in g2c]
        if len(cols) != len(tfset):
            return None, 0
        contain = cg[:, cols].all(1) & screen
        n = int(contain.sum())
        if n < 5:
            return None, n
        d_shared = bln[contain].mean(0) - bg
        d = np.zeros(len(gene_names), np.float32); d[swm] = d_shared
        return d, n

    # ---------- A. alignment spot-check ----------
    print("\n===== A. shared-gene alignment spot-check (Briggs value should land at WM index of SAME gene) =====", flush=True)
    for g in ['GATA2', 'RUNX1', 'PTPRC', 'CD34', 'MYC']:
        if g in shared:
            print(f"  {g}: Briggs shared-pos={shared.index(g)} -> WM idx={g2i[g]} (name@idx='{gene_names[g2i[g]]}') OK={gene_names[g2i[g]]==g}", flush=True)

    # ---------- C. CROSS-DATASET single-TF ----------
    print("\n===== C. CROSS-DATASET single-TF (do datasets agree at k=1?) =====", flush=True)
    drift_j = np.mean([np.asarray(data.de_single[g]['lfc'], np.float32) for g in data.single_split['train']], 0)
    bsingles = {}
    for t in TFS:
        bd, n = briggs_delta([t], bulk_hesc)
        if bd is None:
            print(f"  {t}: no Briggs single (n={n})", flush=True); continue
        bsingles[t] = bd
        j_obs = np.asarray(data.de_single[t]['lfc'], np.float32) if t in data.de_single else None
        wm = wm_delta([g2i[t]])
        c_jb = cc(j_obs, bd, shared_mask) if j_obs is not None else float('nan')
        c_wb = cc(wm, bd, shared_mask)
        c_jb_dr = cc(j_obs - drift_j, bd, shared_mask) if j_obs is not None else float('nan')   # remove Joung drift
        print(f"  {t} (n={n}): corr(JoungObs, Briggs)={c_jb:+.3f} | corr(WMdoop, Briggs)={c_wb:+.3f} "
              f"| Joung-drift-removed vs Briggs={c_jb_dr:+.3f}", flush=True)

    # ---------- D. CROSS-DATASET combo (held-out pairs) + background sensitivity ----------
    print("\n===== D. CROSS-DATASET combo (held-out test pairs) raw + drift-removed, hESC vs Joung-ctrl bg =====", flush=True)
    drift_pred = np.mean([wm_delta([g2i[g]]) for g in data.single_split['train'][:40]], 0)
    for pair in [('TAL1', 'ETV2'), ('LMO2', 'ETV2')]:
        gi = [g2i[t] for t in pair]; pred = wm_delta(gi)
        for bgname, bg in [('hESC', bulk_hesc), ('Joung-ctrl', jbulk)]:
            bd, n = briggs_delta(list(pair), bg)
            if bd is None:
                continue
            c_raw = cc(pred, bd, shared_mask)
            c_dr = cc(pred - drift_pred, bd, shared_mask)               # drift-remove the PREDICTION
            print(f"  {pair} bg={bgname:10s} n={n}: corr(pred,obs) raw={c_raw:+.3f} | pred-drift-removed={c_dr:+.3f}", flush=True)

    # ---------- E. drift dominance ----------
    print("\n===== E. drift dominance (how much of Δ is the shared differentiation program?) =====", flush=True)
    ex_pred = wm_delta([g2i['ETV2'], g2i['TAL1']])
    print(f"  ||drift_pred||/||pred(ETV2,TAL1)|| = {np.linalg.norm(drift_pred)/(np.linalg.norm(ex_pred)+1e-9):.2f}", flush=True)
    if 'ETV2' in bsingles:
        bcombo, _ = briggs_delta(['ETV2', 'TAL1'], bulk_hesc)
        if bcombo is not None:
            drift_b = np.zeros(len(gene_names), np.float32)              # Briggs drift ~ mean single Δ
            drift_b[swm] = np.mean([bsingles[t][swm] for t in bsingles], 0)
            print(f"  ||drift_briggs||/||briggs(ETV2,TAL1)|| = {np.linalg.norm(drift_b)/(np.linalg.norm(bcombo)+1e-9):.2f}", flush=True)
            print(f"  corr(drift_pred, drift_briggs) on shared = {cc(drift_pred, drift_b, shared_mask):+.3f}", flush=True)
    print("\n[audit done]", flush=True)


if __name__ == '__main__':
    main()
