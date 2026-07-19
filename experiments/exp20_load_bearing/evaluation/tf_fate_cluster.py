"""Cluster held-out TFs (n=49) by the top model's per-TF Pnk/allDEG vs the LINEAGE each TF drives —
using the CORRECT HUMAN fetal atlas annotation (Cao 2020, GSE156793, 172 organ-cell-types, cosine,
permutation-FDR): joung_tf_lineage.csv (7 broad lineages L0-L6) + joung_human_annotation_celllevel.csv
(specificity, fdr_qval). NOT the mouse MOCA reference.
Writes results/figures/tf_fate_cluster.png + results/figures/tf_fate_cluster.csv.
"""
import os, sys, numpy as np, torch, scipy.sparse as sp, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import Counter
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
HANN = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/human_dev_annotation"
torch.set_grad_enabled(False)
dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None, effective_tfs=V + "/effective_tf_genes.txt")
de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0); obs = np.stack([np.asarray(de[g], np.float32) for g in te])
Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32); ctrl = np.array(dd['control_idx'])
rng = np.random.default_rng(0); xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), 3000, replace=False)]])
m, gnm, c = load_shift_model('results/grid/v2b_mmd1_deg2/checkpoints/phase1_epoch60.pt', 'cpu'); g2 = {g: i for i, g in enumerate(gnm)}
_, _, eps = m.encoder(xc); A = m.dag.sample_A(1).squeeze(0); zp = m.dag.scm_forward(eps, A); cr = m.decoder(zp).mean(0).cpu().numpy()
pred = np.stack([(m.decoder(m.intervene_shift_mutilate(eps, [g2[g]], A, zp, u_mode='shift', shift_beta=1.0)[0]).mean(0).cpu().numpy() - cr) for g in te])
Pr, Or = pred - drift, obs - drift
adeg, pnk = [], []
for i, g in enumerate(te):
    k = np.ones(len(gn), bool); k[gidx[g]] = False
    pnk.append(np.corrcoef(Pr[i, k], Or[i, k])[0, 1] if Pr[i, k].std() > 1e-9 else np.nan)
    dg = k & (np.abs(Or[i]) > 0.15); adeg.append(np.corrcoef(Pr[i, dg], Or[i, dg])[0, 1] if dg.sum() > 2 and Pr[i, dg].std() > 1e-9 else np.nan)
adeg, pnk = np.array(adeg), np.array(pnk)

# ---- CORRECT human fetal atlas annotation ----
lin, qval, celltype = {}, {}, {}
for r in csv.DictReader(open(f'{HANN}/joung_tf_lineage.csv')):
    lin[r['TF']] = r['lineage_name']; qval[r['TF']] = float(r['fdr_qval']) if r['fdr_qval'] else np.nan; celltype[r['TF']] = r['top_celltype']
spec = {}
for r in csv.DictReader(open(f'{HANN}/joung_human_annotation_celllevel.csv')):
    spec[r['TF']] = float(r['specificity']) if r['specificity'] else np.nan
lineage = np.array([lin.get(g, '(none)') for g in te]); q = np.array([qval.get(g, np.nan) for g in te]); sp_ = np.array([spec.get(g, np.nan) for g in te])
# short lineage label (L#: first token)
def short(s):
    if not s or s == '(none)': return '(none)'
    return s.split(':')[0] + ':' + s.split(':')[1].split('/')[0] if ':' in s else s
slab = np.array([short(x) for x in lineage])
n_ann = int(sum(g in lin for g in te)); n_conf = int(np.nansum(q < 0.10))
print(f"[human-fetal ann] {n_ann}/{len(te)} held-out TFs annotated; {n_conf} confident (FDR q<0.10); {len(set(slab))-1} lineages", flush=True)


def mp(v, mk): x = v[mk]; x = x[np.isfinite(x)]; return (np.mean(x), (np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0)) if len(x) else (np.nan, 0)


print("\n=== held-out TF allDEG by HUMAN-FETAL lineage (Cao 2020) ===")
for L, cnt in Counter(slab).most_common():
    mk = slab == L; a, _ = mp(adeg, mk); p, _ = mp(pnk, mk)
    print(f"  {L:40} n={int(mk.sum()):2}  allDEG {a:+.3f}  Pnk {p:+.3f}", flush=True)
print("\n=== by annotation CONFIDENCE (perm-FDR q<0.10) ===")
conf = q < 0.10
for lab, mk in [('confident (q<0.10)', conf), ('not confident', ~conf & np.isfinite(q)), ('unannotated', ~np.isfinite(q) | np.array([g not in lin for g in te]))]:
    a, ase = mp(adeg, mk); p, _ = mp(pnk, mk)
    print(f"  {lab:22} n={int(mk.sum()):2}  allDEG {a:+.3f}±{ase:.3f}  Pnk {p:+.3f}", flush=True)
fin = np.isfinite(adeg) & np.isfinite(sp_)
if fin.sum() > 2: print(f"  corr(allDEG, cosine specificity) = {np.corrcoef(adeg[fin], sp_[fin])[0,1]:+.3f}", flush=True)

# ---- figure ----
fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6))
present = [L for L, _ in Counter(slab).most_common() if L != '(none)']
pal = ['#2f6db5', '#eb6834', '#1f9e89', '#c0392b', '#6b3fa0', '#d9821e', '#555', '#17a2b8']
COL = {L: pal[i % len(pal)] for i, L in enumerate(present)}
for L in present:
    mk = (slab == L) & np.isfinite(sp_) & np.isfinite(adeg)
    if mk.sum(): ax.scatter(sp_[mk], adeg[mk], s=75, color=COL[L], edgecolor='#333', lw=0.5, alpha=0.9, label=f'{L} (n={int((slab==L).sum())})')
ax.axhline(0, color='#888', lw=0.7); ax.set_xlabel('cosine fate specificity (peak − mean vs 172 fetal cell types)')
ax.set_ylabel('model allDEG (held-out)'); ax.legend(fontsize=7, loc='lower right'); ax.set_title(f'Held-out TFs by human-fetal lineage (Cao 2020, n={n_ann})', fontsize=12)
mu = [mp(adeg, slab == L)[0] for L in present]; se = [mp(adeg, slab == L)[1] for L in present]
bx.bar(range(len(present)), mu, yerr=se, color=[COL[L] for L in present], alpha=0.9, capsize=4)
bx.axhline(0, color='#444', lw=0.7); bx.set_xticks(range(len(present)))
bx.set_xticklabels([f'{L}\n(n={int((slab==L).sum())})' for L in present], fontsize=8, rotation=25, ha='right')
bx.set_ylabel('model allDEG ±SEM'); bx.set_title('allDEG by lineage (human fetal atlas)', fontsize=12)
fig.tight_layout(); out = 'results/figures/tf_fate_cluster.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
with open('results/figures/tf_fate_cluster.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['TF', 'allDEG', 'Pnk', 'lineage', 'top_celltype', 'cosine_specificity', 'fdr_qval'])
    for i, g in enumerate(te): w.writerow([g, f'{adeg[i]:.3f}', f'{pnk[i]:.3f}', lineage[i], celltype.get(g, ''), f'{sp_[i]:.3f}', f'{q[i]:.3f}'])
print('\nwrote', out, '+ tf_fate_cluster.csv (HUMAN FETAL annotation)', flush=True)
