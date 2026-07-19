"""Combo cross-validation eval on the FIXED folds (v2b combo file, seed-42 single split).
Each fold model held out specific combos and is scored ONLY on those (leakage-safe, no seed confound).
  LOPO (40 folds): each pair predicted by the model that left THAT pair out -> pool 40 -> n=40 CV.
  LOTO (10 folds): each model predicts its held-out TF's pairs -> per-fold metric -> mean±SEM over 10.
Reports do_mean vs model-additive vs obs-additive (allDEG, Pnk) + epistasis r_pair, with baselines.
Run: python evaluation/eval_combo_cv.py --device cpu
"""
import os, sys, json, glob, re, argparse, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
T15 = 0.15
torch.set_grad_enabled(False)
F = json.load(open('configs/ablations/combo_folds.json'))
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
train_s = [g for g in dd['pert_split']['train'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in train_s]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc_np = Xc[ctrl[rng.choice(len(ctrl), 2000, replace=False)]]
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
inter = [g for g in gn if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter]); driftm = drift[mask]
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)
tf2ts = {}
for ts in dict.fromkeys(tfset):
    tt = tuple(sorted(t for t in parse_tfs(ts) if t in gidx))
    if len(tt) == 2: tf2ts[tt] = ts


def predict(model, gnm, pairs):
    """return per-pair dict of drift-removed vectors on the intersection: obs, obsadd, do_mean, madd."""
    g2 = {g: i for i, g in enumerate(gnm)}; BETA = 1.0
    xc = torch.tensor(xc_np); _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0)
    zp = model.dag.scm_forward(eps, A); cr = model.decoder(zp).mean(0).cpu().numpy()
    sh = lambda gi: model.compute_shift(torch.tensor([gi])).sum(0)
    def solve(S, u):
        z, _, _ = model.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return model.decoder(z).mean(0).cpu().numpy() - cr
    out = {}
    for p in pairs:
        p = tuple(p)
        if p not in tf2ts or not all(t in g2 for t in p): continue
        mm = tfset == tf2ts[p]
        if mm.sum() < 20: continue
        of = Xk[mm].mean(0) - ref; ob = of[col]
        oa = np.zeros(len(gn), np.float32)
        for g in p: oa += np.asarray(de[g], np.float32)
        Sa, Sb = sh(g2[p[0]]), sh(g2[p[1]]); ua, ub = (BETA * Sa.abs()).clamp(max=1.0), (BETA * Sb.abs()).clamp(max=1.0)
        S = Sa + Sb
        si = model.compute_interaction(torch.tensor([g2[p[0]], g2[p[1]]])) if getattr(model, 'interaction_U', None) is not None else None
        if si is not None: S = S + si
        dm = solve(S, (ua + ub) / 2)[mask]; madd = (solve(Sa, ua) + solve(Sb, ub))[mask]
        out[p] = dict(obs=ob - driftm, obsadd=oa[mask] - driftm, dm=dm, madd=madd)
    return out


def latest(tag):
    cks = glob.glob(f'results/grid/{tag}/checkpoints/phase1_epoch*.pt')
    return max(cks, key=lambda p: int(re.search(r'epoch(\d+)', p).group(1))) if cks else None


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def block_metrics(rows):
    """rows: list of per-pair dicts. across-pair centered per-pair allDEG for do_mean/madd/obsadd + r_pair."""
    O = np.stack([r['obs'] for r in rows]); OA = np.stack([r['obsadd'] for r in rows])
    DM = np.stack([r['dm'] for r in rows]); MA = np.stack([r['madd'] for r in rows]); N = len(rows)
    Od = O - O.mean(0)
    def a15(P):
        Pd = P - P.mean(0); acc = []
        for i in range(N):
            d = np.abs(Od[i]) > T15
            acc.append(np.corrcoef(Pd[i, d], Od[i, d])[0, 1] if d.sum() > 2 and Pd[i, d].std() > 1e-9 else np.nan)
        return np.array(acc)
    dm_a, ma_a, oa_a = a15(DM), a15(MA), a15(OA)
    eps_obs = O - OA; eps_mod = DM - MA; dmask = np.abs(O) > T15
    ns = np.array([eps_obs[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(N)])
    nm = np.array([eps_mod[i][dmask[i]].mean() if dmask[i].sum() else 0.0 for i in range(N)])
    rp = float(np.corrcoef(ns, nm)[0, 1]) if np.std(ns) > 1e-9 and np.std(nm) > 1e-9 else np.nan
    return dm_a, ma_a, oa_a, rp


def msem(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))) if len(v) > 1 else (float(np.mean(v)) if len(v) else np.nan, 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu'); ap.parse_args()
    # ---- LOPO: pool 40 pairs, each from its leave-that-pair-out model ----
    print("=== LOPO (leave-one-pair-out): each pair predicted by the model that left it out ===", flush=True)
    lopo_rows = []
    for name, pairs in F['lopo'].items():
        p = tuple(pairs[0]); tag = f'v2b_cv8_lopo_{name}'; ck = latest(tag)
        if not ck: print(f"  MISSING {tag}", flush=True); continue
        m, gnm, c = load_shift_model(ck, 'cpu'); pr = predict(m, gnm, [p])
        if p in pr: lopo_rows.append(pr[p])
    if len(lopo_rows) >= 2:
        dm, ma, oa, rp = block_metrics(lopo_rows)
        print(f"  n={len(lopo_rows)} pairs", flush=True)
        print(f"  do_mean  allDEG {msem(dm)[0]:.3f}±{msem(dm)[1]:.3f}", flush=True)
        print(f"  model-add allDEG {msem(ma)[0]:.3f}±{msem(ma)[1]:.3f}", flush=True)
        print(f"  obs-add  allDEG {msem(oa)[0]:.3f}±{msem(oa)[1]:.3f}", flush=True)
        print(f"  epistasis r_pair (pooled, n={len(lopo_rows)}) = {rp:+.3f}", flush=True)
    # ---- LOTO: per-fold (each model predicts its held-out TF's pairs), mean±SEM over 10 folds ----
    print("\n=== LOTO (leave-one-TF-out): per-fold do_mean vs baselines, mean±SEM over folds ===", flush=True)
    fold_dm, fold_ma, fold_oa, fold_rp = [], [], [], []
    for tf in F['tfs']:
        tag = f'v2b_cv8_loto_{tf}'; ck = latest(tag)
        if not ck: print(f"  MISSING {tag}", flush=True); continue
        m, gnm, c = load_shift_model(ck, 'cpu'); pr = predict(m, gnm, F['loto'][tf])
        rows = list(pr.values())
        if len(rows) < 2: continue
        dm, ma, oa, rp = block_metrics(rows)
        fold_dm.append(np.nanmean(dm)); fold_ma.append(np.nanmean(ma)); fold_oa.append(np.nanmean(oa)); fold_rp.append(rp)
        print(f"  {tf:8} (n={len(rows)}): do_mean {np.nanmean(dm):.3f}  model-add {np.nanmean(ma):.3f}  obs-add {np.nanmean(oa):.3f}  r_pair {rp:+.3f}", flush=True)
    print(f"\n  LOTO mean±SEM over {len(fold_dm)} folds:", flush=True)
    print(f"    do_mean  allDEG {msem(fold_dm)[0]:.3f}±{msem(fold_dm)[1]:.3f}", flush=True)
    print(f"    model-add allDEG {msem(fold_ma)[0]:.3f}±{msem(fold_ma)[1]:.3f}", flush=True)
    print(f"    obs-add  allDEG {msem(fold_oa)[0]:.3f}±{msem(fold_oa)[1]:.3f}", flush=True)
    print(f"    epistasis r_pair {msem(fold_rp)[0]:+.3f}±{msem(fold_rp)[1]:.3f}", flush=True)
    print("\ndone", flush=True)


if __name__ == '__main__':
    main()
