"""Finer clustering of held-out TFs vs model allDEG, two ways (human fetal, Cao 2020):
 OPT1  finer lineages: Agglomerative(k=14) over the 165 Cao cell-type signatures (double the 7 Leiden
       lineages); each held-out TF inherits its top_celltype's finer lineage.
 OPT2  cluster the held-out TFs directly in 172-dim fetal-cosine space (their full affinity fingerprint)
       into 5 data-driven groups.
Reuses cached per-TF allDEG from results/figures/tf_fate_cluster.csv (no do-op recompute).
Writes results/figures/tf_fate_cluster_fine.png + tf_fate_cluster_fine.csv.
"""
import os, numpy as np, pandas as pd, csv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
H = "/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/analysis/human_dev_annotation"
FIG = "results/figures"
tf = pd.read_csv(f'{FIG}/tf_fate_cluster.csv')                                  # TF, allDEG, Pnk, top_celltype, ...
tf = tf.set_index('TF'); held = list(tf.index)
allDEG = tf['allDEG'].astype(float); topct = tf['top_celltype'].astype(str)


def mp(v): v = np.asarray(v, float); v = v[np.isfinite(v)]; return (v.mean(), v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (v.mean() if len(v) else np.nan, 0)


# ============ OPT1: finer lineages (Agglomerative k=14 over 165 cell-type signatures) ============
sig = pd.read_csv(f'{H}/cao_type_mean_logexpr.csv', index_col=0)                # [165 celltypes x genes]
Z = (sig.values - sig.values.mean(0)) / (sig.values.std(0) + 1e-8)
K1 = 14
lab1 = AgglomerativeClustering(n_clusters=K1).fit_predict(Z)
# name each finer lineage by its 2 most-central cell types (nearest cluster centroid)
ct = list(sig.index); ct2fine = {}
fine_name = {}
for k in range(K1):
    idx = np.where(lab1 == k)[0]; cen = Z[idx].mean(0)
    order = idx[np.argsort(((Z[idx] - cen) ** 2).sum(1))]
    fine_name[k] = 'F%d:' % k + '/'.join([ct[i].split('-')[-1][:16] for i in order[:2]])
    for i in idx: ct2fine[ct[i]] = fine_name[k]
tf_fine = np.array([ct2fine.get(c, '(none)') for c in topct])
print(f"OPT1 — {K1} finer lineages (Agglomerative on 165 signatures). Held-out allDEG by finer lineage:")
rows1 = []
for f in sorted(set(tf_fine), key=lambda f: -mp(allDEG[tf_fine == f])[0]):
    mk = tf_fine == f; mu, se = mp(allDEG[mk])
    if mk.sum() >= 1: print(f"  {f:34} n={int(mk.sum()):2}  allDEG {mu:+.3f}±{se:.3f}"); rows1.append((f, int(mk.sum()), mu, se))

# ============ OPT2: cluster held-out TFs in 172-dim fetal-cosine space ============
cos = pd.read_csv(f'{H}/joung_human_cosine.csv', index_col=0)                   # [1084 TFs x 172 celltypes]
common = [g for g in held if g in cos.index]
Xtf = cos.loc[common].values                                                   # [n x 172]
K2 = 5
lab2 = KMeans(n_clusters=K2, n_init=10, random_state=0).fit_predict(Xtf)
cl = pd.Series(lab2, index=common)
# name each TF-cluster by its top-2 mean-affinity cell types
types172 = list(cos.columns)
cl_name = {}
for k in range(K2):
    mk = cl.values == k; prof = Xtf[mk].mean(0); top = np.argsort(-prof)[:2]
    cl_name[k] = 'C%d:' % k + '/'.join([types172[i].split('-')[-1][:16] for i in top])
tfcl = pd.Series([cl_name[cl[g]] for g in common], index=common)
print(f"\nOPT2 — {K2} data-driven TF clusters in 172-dim fetal-cosine space. allDEG by cluster:")
rows2 = []
for k in range(K2):
    mem = [g for g in common if cl[g] == k]; a = allDEG.loc[mem]; mu, se = mp(a.values)
    print(f"  {cl_name[k]:34} n={len(mem):2}  allDEG {mu:+.3f}±{se:.3f}  e.g. {mem[:4]}"); rows2.append((cl_name[k], len(mem), mu, se))

# ============ figure ============
fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 6))
r1 = [r for r in rows1 if r[1] >= 2]; r1.sort(key=lambda r: -r[2])
ax.barh(range(len(r1)), [r[2] for r in r1], xerr=[r[3] for r in r1], color='#2f6db5', alpha=0.9, capsize=3)
ax.set_yticks(range(len(r1))); ax.set_yticklabels([f'{r[0]} (n={r[1]})' for r in r1], fontsize=8); ax.invert_yaxis()
ax.axvline(0, color='#444', lw=0.7); ax.set_xlabel('model allDEG ±SEM'); ax.set_title(f'OPT1: {K1} finer lineages (Agglomerative on 165 cell types)', fontsize=12)
r2 = sorted(rows2, key=lambda r: -r[2])
bx.barh(range(len(r2)), [r[2] for r in r2], xerr=[r[3] for r in r2], color='#1f9e89', alpha=0.9, capsize=3)
bx.set_yticks(range(len(r2))); bx.set_yticklabels([f'{r[0]} (n={r[1]})' for r in r2], fontsize=9); bx.invert_yaxis()
bx.axvline(0, color='#444', lw=0.7); bx.set_xlabel('model allDEG ±SEM'); bx.set_title(f'OPT2: {K2} TF clusters in 172-dim fetal-cosine space', fontsize=12)
fig.tight_layout(); out = f'{FIG}/tf_fate_cluster_fine.png'; fig.savefig(out, dpi=150, bbox_inches='tight')
with open(f'{FIG}/tf_fate_cluster_fine.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['TF', 'allDEG', 'top_celltype', 'finer_lineage(OPT1)', 'tf_cluster(OPT2)'])
    for g in held: w.writerow([g, f'{allDEG[g]:.3f}', topct[g], ct2fine.get(topct[g], '(none)'), (cl_name[cl[g]] if g in common else '(none)')])
print('\nwrote', out, '+ tf_fate_cluster_fine.csv', flush=True)
