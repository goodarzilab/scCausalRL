#!/bin/bash
#SBATCH --job-name=cao_loom_inspect
#SBATCH --partition=cpu_high_mem,cpu_batch,cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=logs/loom_inspect_%j.out
#SBATCH --error=logs/loom_inspect_%j.err
set -euo pipefail
D=/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal
cd "$D"
echo "start $(date) on $(hostname)"
[ -f GSE156793_S3_gene_count.loom ] || { echo "decompressing..."; gunzip -kf GSE156793_S3_gene_count.loom.gz; }
ls -la --block-size=G GSE156793_S3_gene_count.loom
python - <<'PY'
import loompy, numpy as np, pandas as pd, h5py
D='/large_storage/goodarzilab/shervin/CausalBioRL/Cao2020_Human_Fetal'
JV='/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
# Joung gene symbols (read var index only, no X)
with h5py.File(JV,'r') as f:
    vn = f['var']['_index'][:] if 'var' in f and '_index' in f['var'] else f['var/_index'][:]
    joung = set(str(x.decode() if isinstance(x,bytes) else x).upper() for x in vn)
print('Joung genes:', len(joung))
s1 = pd.read_csv(f'{D}/GSE156793_S1_metadata_cells.txt.gz', usecols=['sample','Organ_cell_lineage','Main_cluster_name'])
samp = set(s1['sample'].astype(str).values[:200000])
print('Organ_cell_lineage n types:', s1['Organ_cell_lineage'].nunique(), '| Main_cluster_name n types:', s1['Main_cluster_name'].nunique())
with loompy.connect(f'{D}/GSE156793_S3_gene_count.loom','r') as ds:
    print('shape (genes x cells):', ds.shape)
    print('ROW attrs:', list(ds.ra.keys())); print('COL attrs:', list(ds.ca.keys()))
    for k in ds.ca.keys():
        v = np.array([str(x) for x in ds.ca[k][:5000]])
        ov = len(set(v) & samp)
        print(f"  ca[{k}] sample={v[0]!r} overlap_with_S1_sample(5k)={ov}")
    for k in ds.ra.keys():
        v = np.array([str(x).upper() for x in ds.ra[k][:]])
        ov = len(set(v) & joung)
        print(f"  ra[{k}] sample={ds.ra[k][0]!r} overlap_with_Joung_genes={ov}")
PY
echo "done $(date)"
