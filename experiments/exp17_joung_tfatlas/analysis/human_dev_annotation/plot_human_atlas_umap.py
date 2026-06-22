"""UMAP of the Cao 2020 human fetal cell atlas (for the slide deck), colored by Organ.
Subsamples cells from S1_metadata_cells (Global_umap_1/2) to keep it light.

Run:  python analysis/human_dev_annotation/plot_human_atlas_umap.py
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REF_DIR = '/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
S1 = os.path.join(REF_DIR, 'GSE156793_S1_metadata_cells.txt.gz')
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'figures', 'joung'))
os.makedirs(OUT, exist_ok=True)

cols = ['Organ', 'Main_cluster_name', 'Global_umap_1', 'Global_umap_2']
print('reading S1 metadata (subsampled)...', flush=True)
df = pd.read_csv(S1, sep=None, engine='python', usecols=lambda c: c in cols)
df = df.dropna(subset=['Global_umap_1', 'Global_umap_2'])
if len(df) > 200000:
    df = df.sample(200000, random_state=0)
organs = sorted(df['Organ'].dropna().unique())
cmap = plt.get_cmap('tab20', len(organs))
fig, ax = plt.subplots(figsize=(9, 8))
for i, org in enumerate(organs):
    s = df[df['Organ'] == org]
    ax.scatter(s['Global_umap_1'], s['Global_umap_2'], s=1.2, alpha=0.4, color=cmap(i), label=org, rasterized=True)
ax.set_title(f'Cao 2020 human fetal atlas — {len(df):,} cells (subsampled), {len(organs)} organs', fontsize=11)
ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_xticks([]); ax.set_yticks([])
ax.legend(markerscale=6, fontsize=7, ncol=2, loc='upper right', frameon=False)
fig.tight_layout()
p = os.path.join(OUT, 'human_fetal_atlas_umap.png'); fig.savefig(p, dpi=140); print('saved', p)
