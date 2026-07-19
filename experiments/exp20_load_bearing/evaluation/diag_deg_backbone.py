"""Diagnose WHY the deg5 backbone gives better singles but worse combos than deg2, under the frozen-head
regime. For each backbone: single-shift magnitude over held-out TFs, combo u-saturation (fraction of
latent programs fully clamped u>0.99 → those A-rows detached), and the learned interaction norm.
Hypothesis: higher DEG-MSE weight -> larger shifts -> more u-saturation on combos -> more DAG
detachment -> worse combo composition. Also reports per-combo model-additive vs do_mean allDEG.
"""
import os, sys, glob, re, numpy as np, torch, scipy.sparse as sp, anndata as ad
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.combo_eval_v2b import parse_tfs, COMBO
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 800, replace=False)]])
C = ad.read_h5ad(COMBO); Xk = C.X; Xk = (Xk.toarray() if sp.issparse(Xk) else np.asarray(Xk)).astype(np.float32)
Xk = np.log1p(Xk / (Xk.sum(1, keepdims=True) + 1e-9) * 1e4); cvar = {g: i for i, g in enumerate(map(str, C.var_names))}
tfset = C.obs['TF_set'].astype(str).values; ref = Xk.mean(0)

for tag, src in [('deg2 (fthead_r32)', 'v2b_fthead_r32'), ('deg5 (deg5_frozen_r32)', 'v2b_deg5_frozen_r32')]:
    ck = max(glob.glob(f'results/grid/{src}/checkpoints/phase1_epoch*.pt'), key=lambda p: int(re.search(r'epoch(\d+)', p).group(1)))
    m, gnm, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    BETA = float(c.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))
    held = {tuple(sorted(map(str, k))) for k in (c.get('config', {}).get('phase1', {}).get('combo_test_keys') or [])}
    inter = [g for g in gnm if g in cvar]; col = np.array([cvar[g] for g in inter]); mask = np.array([gidx[g] for g in inter]); driftm = drift[mask]
    _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    def solve(S, u):
        z, _, _ = m.dag.scm_intervene_u(eps, A, eps + S.unsqueeze(0), u); return m.decoder(z).mean(0).cpu().numpy() - cr
    sn = [float(m.compute_shift(torch.tensor([g2[g]])).sum(0).norm()) for g in te if g in g2]
    sats, umean, sint, madd_a, dm_a = [], [], [], [], []
    for ts in dict.fromkeys(tfset):
        tfs = [t for t in parse_tfs(ts) if t in g2]
        if len(tfs) != 2 or tuple(sorted(tfs)) not in held: continue
        mm = tfset == ts
        if mm.sum() < 20: continue
        Sa = m.compute_shift(torch.tensor([g2[tfs[0]]])).sum(0); Sb = m.compute_shift(torch.tensor([g2[tfs[1]]])).sum(0)
        ua = (BETA * Sa.abs()).clamp(max=1.0); ub = (BETA * Sb.abs()).clamp(max=1.0); u = (ua + ub) / 2
        sats.append(float((u > 0.99).float().mean())); umean.append(float(u.mean()))
        si = m.compute_interaction(torch.tensor([g2[tfs[0]], g2[tfs[1]]])); sint.append(float(si.norm()) if si is not None else 0.0)
        of = Xk[mm].mean(0) - ref; obs = of[col]
        S = Sa + Sb + (si if si is not None else 0)
        dm = solve(S, u)[mask]; madd = (solve(Sa, ua) + solve(Sb, ub))[mask]
        Or = obs - driftm
        dg = np.abs(Or) > 0.15
        def adeg(P):
            Pr = P - driftm; return np.corrcoef(Pr[dg], Or[dg])[0, 1] if dg.sum() > 2 and Pr[dg].std() > 1e-9 else np.nan
        madd_a.append(adeg(madd)); dm_a.append(adeg(dm))
    print(f"\n{tag}  n_held_combos={len(sats)}", flush=True)
    print(f"  single-shift ||S|| over held-out TFs : mean {np.mean(sn):.2f}  median {np.median(sn):.2f}  max {np.max(sn):.2f}", flush=True)
    print(f"  combo u        : mean {np.mean(umean):.3f}   frac programs FULLY clamped (u>0.99): {np.mean(sats):.3f}", flush=True)
    print(f"  learned ||S_int|| (mean over combos) : {np.mean(sint):.3f}", flush=True)
    print(f"  model-additive allDEG : {np.nanmean(madd_a):.3f}    do_mean allDEG : {np.nanmean(dm_a):.3f}    head gain: {np.nanmean(dm_a)-np.nanmean(madd_a):+.3f}", flush=True)
