"""Re-score the WHOLE v2b sweep with the CORRECT A_gap: the correctness GAIN of the learned A over a
SHUFFLED-A null (magnitude-matched, wiring scrambled) on held-out drift-removed metrics — NOT the old
cos(Δ_Aon,Δ_Aoff) which only measures whether A moves the output (and ranks arms backwards).
Per arm: disc/Pnk/allDEG under real A, shuffled A (mean of 2 seeds), zero A; and real-shuffled gain.
Writes results/figures/agap_correctness.csv.  Run detached: python evaluation/agap_correctness.py
"""
import os, sys, glob, numpy as np, torch, scipy.sparse as sp, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
obs = np.stack([np.asarray(de[g], np.float32) for g in te])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 1500, replace=False)]])


def zr(M): M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def suite(P):
    Pr, Or = P - drift, obs - drift
    C = zr(Pr) @ zr(Or).T; d = np.diag(C)[:, None]; T = len(P); disc = float(((d > C).sum(1) / (T - 1)).mean())
    pnk, adeg = [], []
    for i, g in enumerate(te):
        keep = np.ones(P.shape[1], bool); keep[gidx[g]] = False
        x, y = Pr[i, keep], Or[i, keep]; pnk.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 else np.nan)
        deg = keep & (np.abs(Or[i]) > 0.15); xd, yd = Pr[i, deg], Or[i, deg]
        adeg.append(np.corrcoef(xd, yd)[0, 1] if deg.sum() > 2 and xd.std() > 1e-9 else np.nan)
    return np.array([disc, float(np.nanmean(pnk)), float(np.nanmean(adeg))])


rows = []
arms = sorted(glob.glob('results/grid/v2b_*/checkpoints/phase1_epoch60.pt'))
print(f"{len(arms)} arms\n{'arm':30}{'disc_r':>7}{'Pnk_r':>7}{'aDEG_r':>7} | {'dPnk':>7}{'daDEG':>7}{'ddisc':>7}", flush=True)
for ck in arms:
    tag = ck.split('/')[2]
    try:
        m, gnm, c = load_shift_model(ck, 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}; _, _, eps = m.encoder(xc)
        um = c.get('config', {}).get('phase1', {}).get('pert_mutilate_u_mode', 'shift')
        bt = float(c.get('config', {}).get('phase1', {}).get('pert_mutilate_beta', 1.0))

        def pset(A):
            zp = m.dag.scm_forward(eps, A)
            return np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode=um, shift_beta=bt)[0]).mean(0).cpu().numpy() - m.decoder(zp).mean(0).cpu().numpy()) for g in te])
        m.dag.disabled = False; Ar = m.dag.sample_A(1).squeeze(0)
        sr = suite(pset(Ar))
        shf = []
        for s in range(2):
            flat = Ar.flatten(); perm = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(s))
            Ash = flat[perm].reshape_as(Ar).clone(); Ash.fill_diagonal_(0.0)
            shf.append(suite(pset(m.dag._spectral_clamp(Ash.unsqueeze(0)).squeeze(0))))
        sh = np.mean(shf, 0)
        gain = sr - sh
        rows.append(dict(arm=tag, disc_real=sr[0], Pnk_real=sr[1], aDEG_real=sr[2],
                         disc_shuf=sh[0], Pnk_shuf=sh[1], aDEG_shuf=sh[2],
                         dPnk=gain[1], daDEG=gain[2], ddisc=gain[0]))
        print(f"{tag:30}{sr[0]:>7.3f}{sr[1]:>7.3f}{sr[2]:>7.3f} | {gain[1]:>+7.3f}{gain[2]:>+7.3f}{gain[0]:>+7.3f}", flush=True)
    except Exception as e:
        print(f"{tag:30} FAIL {type(e).__name__}: {e}", flush=True)
if rows:
    out = 'results/figures/agap_correctness.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('\nwrote', out, flush=True)
