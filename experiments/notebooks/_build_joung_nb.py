"""Assemble the Joung TF Atlas analysis notebook from cell sources (nbformat)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Joung TF Atlas — exploratory data analysis
Single-cell analysis of the **Joung et al. 2023** TF-overexpression atlas (GSE216481), as
preprocessed for exp17 (`joung_singles_proc.h5ad` = single-TF cells, `joung_combo_proc.h5ad`
= combos/triples), on the shared 9,923-gene axis. Figures saved to
`experiments/figures/joung/`. We subsample for UMAP tractability (the full singles set is
1.14M cells).""")

code("""import os, numpy as np, pandas as pd, scanpy as sc, matplotlib.pyplot as plt
sc.settings.verbosity = 1
FIGDIR = os.path.abspath('../figures/joung'); os.makedirs(FIGDIR, exist_ok=True)
sc.settings.figdir = FIGDIR
DATA = os.path.abspath('../exp17_joung_tfatlas/data')
SINGLES = f'{DATA}/joung_singles_proc.h5ad'; COMBO = f'{DATA}/joung_combo_proc.h5ad'
rng = np.random.default_rng(0)
print('singles:', SINGLES); print('combo  :', COMBO)""")

md("## 1. Dataset overview + TF cell-count distribution")
code("""import anndata as ad
a = ad.read_h5ad(SINGLES, backed='r')
obs = a.obs.copy()
print('singles shape:', a.shape)
print('control cells:', int(obs['is_control'].sum()),
      '| real-TF cells:', int((~obs['is_control']).sum()),
      '| unique TFs:', obs.loc[~obs['is_control'],'TF_gene'].nunique())
vc = obs.loc[~obs['is_control'],'TF_gene'].value_counts()
print('cells/TF: median=%.0f mean=%.0f min=%d max=%d' % (vc.median(), vc.mean(), vc.min(), vc.max()))
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].hist(vc.values, bins=60, color='#4C72B0'); ax[0].set_xlabel('cells per TF'); ax[0].set_ylabel('# TFs')
ax[0].set_title('TF cell-count distribution (singles)'); ax[0].set_yscale('log')
top = vc.head(20)[::-1]
ax[1].barh(range(len(top)), top.values, color='#55A868'); ax[1].set_yticks(range(len(top)))
ax[1].set_yticklabels(top.index, fontsize=7); ax[1].set_title('Top-20 TFs by #cells'); ax[1].set_xlabel('#cells')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/tf_cell_counts.png', dpi=130, bbox_inches='tight'); plt.show()
display(vc.head(10).to_frame('n_cells'))""")

md("## 2. Subsample + standard scanpy pipeline (normalize → HVG → PCA → UMAP → Leiden)")
code("""# stratified subsample: control sample + up to K cells/TF
ctrl_idx = np.where(obs['is_control'].values)[0]
keep = list(rng.choice(ctrl_idx, min(6000, len(ctrl_idx)), replace=False))
K = 25
for g, sub in obs.loc[~obs['is_control']].groupby('TF_gene'):
    ii = sub.index.map(obs.index.get_loc).values
    keep += list(ii if len(ii)<=K else rng.choice(ii, K, replace=False))
keep = np.sort(np.array(keep))
print('subsample:', len(keep), 'cells')
sub = a[keep].to_memory()
sub.layers['counts'] = sub.X.copy()
sc.pp.normalize_total(sub, target_sum=1e4); sc.pp.log1p(sub)
sc.pp.highly_variable_genes(sub, n_top_genes=2000)
sub.raw = sub
sc.pp.pca(sub, n_comps=50)
sc.pp.neighbors(sub, n_neighbors=15, n_pcs=50)
sc.tl.umap(sub)
try:
    sc.tl.leiden(sub, resolution=1.0, flavor='igraph', n_iterations=2, directed=False)
except Exception:
    sc.tl.leiden(sub, resolution=1.0)
print('UMAP + Leiden done; clusters:', sub.obs['leiden'].nunique())""")

md("## 3. UMAP — control vs TF-overexpression + Leiden structure")
code("""sub.obs['compartment'] = np.where(sub.obs['is_control'].values, 'control (GFP/mCherry)', 'TF-OE')
fig, ax = plt.subplots(1,2, figsize=(13,5))
sc.pl.umap(sub, color='compartment', ax=ax[0], show=False, title='control vs TF-overexpression', size=6)
sc.pl.umap(sub, color='leiden', ax=ax[1], show=False, title='Leiden clusters', size=6, legend_loc='on data')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/umap_compartment_leiden.png', dpi=130, bbox_inches='tight'); plt.show()""")

md("## 4. UMAP colored by lineage marker genes + key differentiation TFs")
code("""# pluripotency + germ-layer / lineage markers (those present on the axis)
markers = [g for g in ['POU5F1','NANOG','SOX2','TBXT','SOX17','FOXA2','PAX6','SOX1','GATA4',
                       'HAND1','MSGN1','CDX2','TBX6','MYOD1','NEUROD1'] if g in sub.var_names]
sc.pl.umap(sub, color=markers, ncols=5, show=False, size=6, color_map='viridis')
plt.savefig(f'{FIGDIR}/umap_markers.png', dpi=130, bbox_inches='tight'); plt.show()
print('markers shown:', markers)""")

md("## 5. Which TFs reshape the transcriptome most? (impact vs control)")
code("""ctrl_mask = sub.obs['is_control'].values
Xc = sub.raw.X[ctrl_mask]; ctrl_mean = np.asarray(Xc.mean(0)).ravel()
rows = []
import scipy.sparse as sp
Xr = sub.raw.X
for g, idx in sub.obs.loc[~sub.obs['is_control']].groupby('TF_gene').groups.items():
    rr = sub.obs.index.get_indexer(idx)
    if len(rr) < 8: continue
    m = np.asarray(Xr[rr].mean(0)).ravel() - ctrl_mean
    rows.append({'TF': g, 'n_cells': len(rr), 'L2_impact': float(np.linalg.norm(m)),
                 'n_up': int((m>0.5).sum()), 'n_down': int((m<-0.5).sum())})
imp = pd.DataFrame(rows).sort_values('L2_impact', ascending=False)
fig, ax = plt.subplots(figsize=(9,5)); t = imp.head(25)[::-1]
ax.barh(range(len(t)), t['L2_impact'], color='#C44E52'); ax.set_yticks(range(len(t)))
ax.set_yticklabels(t['TF'], fontsize=7); ax.set_xlabel('‖mean expr shift vs control‖₂')
ax.set_title('Top-25 highest-impact TFs (transcriptome remodeling)')
plt.tight_layout(); plt.savefig(f'{FIGDIR}/tf_impact.png', dpi=130, bbox_inches='tight'); plt.show()
display(imp.head(15).reset_index(drop=True))""")

md("## 6. Combinatorial screen — pairs + triples (held-out eval set)")
code("""cb = ad.read_h5ad(COMBO)
print('combinatorial shape:', cb.shape)
print('set sizes:', cb.obs['set_size'].value_counts().to_dict())
sets = cb.obs['TF_set'].value_counts()
print('unique TF sets:', len(sets))
display(sets.head(20).to_frame('n_cells'))
# subsample + UMAP
ci = rng.choice(cb.n_obs, min(40000, cb.n_obs), replace=False)
cs = cb[np.sort(ci)].copy()
sc.pp.normalize_total(cs, target_sum=1e4); sc.pp.log1p(cs)
sc.pp.highly_variable_genes(cs, n_top_genes=2000)
sc.pp.pca(cs, n_comps=50); sc.pp.neighbors(cs, n_pcs=50); sc.tl.umap(cs)
cs.obs['set_size'] = cs.obs['set_size'].astype(str)
# label top combos, rest 'other'
topsets = set(sets.head(10).index)
cs.obs['combo'] = np.where(cs.obs['TF_set'].isin(topsets), cs.obs['TF_set'].astype(str), 'other')
fig, ax = plt.subplots(1,2, figsize=(15,5.5))
sc.pl.umap(cs, color='set_size', ax=ax[0], show=False, title='pair (2) vs triple (3) TF combos', size=8)
sc.pl.umap(cs, color='combo', ax=ax[1], show=False, title='top-10 TF combinations', size=8)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/umap_combos.png', dpi=130, bbox_inches='tight'); plt.show()""")

code("""# the held-out eval sets: real pairs + triples
def real_tfs(s): return [t.split('-')[-1] for t in s.split(',')]
pairs = sorted({s for s in cb.obs['TF_set'].unique() if len(real_tfs(s))==2 and 'GFP' not in s and 'mCherry' not in s})
triples = sorted({s for s in cb.obs['TF_set'].unique() if len(real_tfs(s))>=3})
print('real pairs (%d):' % len(pairs)); [print('  ', '+'.join(real_tfs(p))) for p in pairs[:50]]
print('triples (%d):' % len(triples)); [print('  ', '+'.join(real_tfs(t))) for t in triples]""")

md("""## Summary
- Singles atlas: 1.14M cells, ~1,769 TFs (median ~55 cells/TF) + GFP/mCherry control on a
  9,923-gene axis. UMAP shows control cells form a tight pluripotent core; high-impact TFs
  drive cells into distinct differentiation states.
- Combinatorial: 167,947 cells, 44 real TF pairs + 3 triples — the held-out compositional eval
  for the exp17 world model (additive vs mutilation prediction at k=2 and k=3).
- All figures in `experiments/figures/joung/`.""")

nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                  'language_info': {'name': 'python'}}
out = 'joung_tf_atlas_analysis.ipynb'
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
