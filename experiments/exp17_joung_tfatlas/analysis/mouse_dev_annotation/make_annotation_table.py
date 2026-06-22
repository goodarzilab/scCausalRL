"""Merge the two tier annotations into one confidence-gated developmental-annotation
table for the Joung TF atlas.

Combines per-TF calls from the coarse (Main_cell_type, early-only) and fine
(Sub_trajectory_name) references, attaches OE cell counts, and assigns a confidence tier
from the within-tier specificity percentile. Output: joung_tf_developmental_annotation.csv
(one row per annotated TF, n≥50 cells).
"""
import os
import numpy as np
import pandas as pd
import anndata as ad

HERE = os.path.dirname(os.path.abspath(__file__))
JOUNG = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def cell_counts():
    a = ad.read_h5ad(JOUNG, backed='r')
    tf = a.obs['TF_gene'].astype(str)
    ctrl = a.obs['is_control'].values if 'is_control' in a.obs else np.zeros(len(tf), bool)
    return tf[~np.asarray(ctrl, bool)].value_counts().to_dict()


def load_tier(prefix):
    d = pd.read_csv(os.path.join(HERE, f'{prefix}_annotation.csv'))
    d['TF'] = d['TF'].astype(str)
    d['pct'] = d['specificity'].rank(pct=True)                 # within-tier percentile
    return d.set_index('TF')


def main():
    nc = cell_counts()
    main = load_tier('joung_devannot_mouse_ref_Main_cell_type')
    sub = load_tier('joung_devannot_mouse_ref_Sub_trajectory_name')
    tfs = sorted(set(main.index))                              # annotated TFs (n≥50)

    rows = []
    for tf in tfs:
        m = main.loc[tf]; s = sub.loc[tf] if tf in sub.index else None
        # confidence from the (finer) sub tier percentile, falling back to main
        pct = (s['pct'] if s is not None else m['pct'])
        conf = 'high' if pct >= 0.90 else 'medium' if pct >= 0.75 else 'low'
        rows.append({
            'TF': tf,
            'n_cells': int(nc.get(tf, 0)),
            'confidence': conf,
            'main_fate': m['top_celltype'],
            'main_specificity': round(float(m['specificity']), 3),
            'sub_fate': s['top_celltype'] if s is not None else 'NA',
            'sub_specificity': round(float(s['specificity']), 3) if s is not None else np.nan,
            'main_top3': m['annotation'],
            'sub_top3': s['annotation'] if s is not None else 'NA',
        })
    df = pd.DataFrame(rows)
    order = {'high': 0, 'medium': 1, 'low': 2}
    df = df.sort_values(['confidence', 'sub_specificity'],
                        key=lambda c: c.map(order) if c.name == 'confidence' else -c)
    out = os.path.join(HERE, 'joung_tf_developmental_annotation.csv')
    df.to_csv(out, index=False)
    print(f'saved → {out}')
    print(f'\n{len(df)} TFs annotated (n≥50 cells)')
    print('confidence tiers:', df['confidence'].value_counts().to_dict())
    print('\n=== HIGH-confidence calls (top sub-trajectory specificity decile) ===')
    pd.set_option('display.max_colwidth', 40); pd.set_option('display.width', 160)
    print(df[df.confidence == 'high'][['TF', 'n_cells', 'main_fate', 'sub_fate',
          'sub_specificity']].head(30).to_string(index=False))


if __name__ == '__main__':
    main()
