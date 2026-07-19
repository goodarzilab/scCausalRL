"""Separated Joung-only vs Joung+Norman tables + two figures:
 (A) COMBOS (held-out Joung combos, n=11): do_mean vs model-additive vs obs-additive allDEG + r_pair,
     combo-trained arms (Joung-only & JN) + the singles-trained top model (n=40).
 (B) SINGLES-HURT check (held-out singles, n=49): does combo/joint training degrade singles allDEG?
Singles recomputed per-TF for error bars; combo numbers from the completed eval (with their SEM).
Writes results/figures/combo_vs_singles.png.
"""
import os, sys, numpy as np, torch, scipy.sparse as sp
sys.path.insert(0, '.')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0); obs = np.stack([np.asarray(de[g], np.float32) for g in te])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 2500, replace=False)]])


def singles_allDEG(tag):
    m, gnm, c = load_shift_model(f'results/grid/{tag}/checkpoints/phase1_epoch60.pt', 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
    _, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
    P = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
    Pr, Or = P - drift, obs - drift; ad = []
    for i, g in enumerate(te):
        k = np.ones(len(gn), bool); k[gidx[g]] = False; dg = k & (np.abs(Or[i]) > 0.15)
        ad.append(np.corrcoef(Pr[i, dg], Or[i, dg])[0, 1] if dg.sum() > 2 and Pr[i, dg].std() > 1e-9 else np.nan)
    ad = np.array(ad); return np.nanmean(ad), np.nanstd(ad, ddof=1) / np.sqrt(np.isfinite(ad).sum())


# singles arms: (label, tag, group)
SARMS = [('mmd1_deg2\n(singles-only)', 'v2b_mmd1_deg2', 'J'), ('mmd1 base\n(singles-only)', 'v2b_masked_tgt_delta_plus_mmd', 'J'),
         ('deg2_Jcombo\n(combo)', 'v2b_mmd1_deg2_Jcombo_ho50', 'J'), ('mmd1_Jcombo\n(combo)', 'v2b_mmd1_Jcombo_ho50', 'J'),
         ('deg2_JN\n(singles,joint)', 'v2b_mmd1_deg2_JN', 'JN'), ('deg2_JNcombo\n(combo,joint)', 'v2b_mmd1_deg2_JNcombo_ho50', 'JN'),
         ('mmd1_JNcombo\n(combo,joint)', 'v2b_mmd1_JNcombo_ho50', 'JN')]
S = {}
for lab, tag, grp in SARMS:
    S[lab] = (*singles_allDEG(tag), grp); print(f"singles {lab.splitlines()[0]:16} allDEG {S[lab][0]:+.3f}±{S[lab][1]:.3f}", flush=True)

# combo numbers from the completed comprehensive eval (held-out Joung combos)
# label: (n, do_mean, do_mean_sem, model_add, obs_add, r_pair, group)
C = {
    'mmd1_deg2\n(singles, n=40)': (40, 0.180, 0.042, 0.181, 0.191, -0.01, 'J'),
    'mmd1_Jcombo\n(n=11)':        (11, 0.558, 0.067, 0.481, 0.311, +0.914, 'J'),
    'deg2_Jcombo\n(n=11)':        (11, 0.549, 0.070, 0.482, 0.311, +0.923, 'J'),
    'mmd1_JNcombo\n(n=11)':       (11, 0.541, 0.066, 0.476, 0.311, +0.875, 'JN'),
    'deg2_JNcombo\n(n=11)':       (11, 0.572, 0.068, 0.506, 0.311, +0.877, 'JN'),
}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.5))
# ---- Panel A: combos ----
labs = list(C); x = np.arange(len(labs)); w = 0.38
a1.bar(x - w/2, [C[l][1] for l in labs], w, yerr=[C[l][2] for l in labs], color='#eb6834', capsize=3, label='do_mean (do-op)', alpha=0.92)
a1.bar(x + w/2, [C[l][3] for l in labs], w, color='#1f9e89', capsize=3, label='model-additive', alpha=0.92)
# obs-additive = fixed no-epistasis baseline (0.311 on the n=11 held-out combos) — drawn as a CONSTANT line
a1.axhline(0.311, ls='--', lw=1.6, color='#8a8a8a', label='obs-additive baseline (0.311, n=11 combos)')
for i, l in enumerate(labs): a1.text(i, max(C[l][1], C[l][3]) + 0.03, f"r={C[l][5]:+.2f}", ha='center', fontsize=9, fontweight='bold', color='#4a3aa7')
a1.axhline(0, color='#444', lw=0.7); a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=8)
a1.set_ylabel('combo allDEG ±SEM (held-out JOUNG combos)'); a1.legend(fontsize=9, loc='upper left')
a1.set_title('COMBOS: do_mean beats additive & predicts epistasis (r) — only when combo-trained', fontsize=11)
# ---- Panel B: singles-hurt ----
slabs = [l for l, _, _ in SARMS]; xb = np.arange(len(slabs))
cols = ['#2f6db5' if 'singles-only' in l or 'singles,joint' in l else '#c0392b' for l in slabs]
a2.bar(xb, [S[l][0] for l in slabs], yerr=[S[l][1] for l in slabs], color=cols, capsize=3, alpha=0.92)
a2.axhline(0.31, ls='--', color='#888', lw=1.2, label='ESM-kNN copy (0.31)')
a2.axhline(0, color='#444', lw=0.7); a2.set_xticks(xb); a2.set_xticklabels(slabs, fontsize=8, rotation=12, ha='right')
a2.set_ylabel('singles allDEG ±SEM (held-out singles, n=49)'); a2.legend(fontsize=9)
a2.set_title('SINGLES: combo/joint training DEGRADES singles (blue=singles-only, red=combo-trained)', fontsize=11)
fig.tight_layout(); out = 'results/figures/combo_vs_singles.png'; fig.savefig(out, dpi=150, bbox_inches='tight'); print('wrote', out, flush=True)
