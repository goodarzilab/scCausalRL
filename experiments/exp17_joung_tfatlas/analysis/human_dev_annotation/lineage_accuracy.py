"""SUPERVISED validation of the TF→fate annotation: for a curated set of master-regulator TFs
with KNOWN germ-layer/lineage, how accurately does our shift-cosine annotation predict the
lineage? (Complements the unsupervised permutation-FDR, which measures confidence, not correctness.)

Uses the saved TF×172-cell-type cosine matrix (joung_human_cosine.csv) — light, no raw data.
Predicts lineage two ways: top-cell-type→lineage, and lineage-aggregated (mean cosine over each
lineage's cell types). Reports accuracy overall + among FDR-confident TFs + confusion matrix.

Run:  python analysis/human_dev_annotation/lineage_accuracy.py
"""
import os, re, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'figures', 'joung'))
COS = f'{HERE}/joung_human_cosine.csv'; ANN = f'{HERE}/joung_human_annotation.csv'

# ── coarse lineage map over the 172 "Organ-CellType" names (keyword rules, first match wins) ──
def lineage_of(ct):
    s = ct.lower()
    if any(k in s for k in ['trophoblast', 'syncytio', 'cytotrophoblast']): return 'TROPHOBLAST'
    if s.startswith('eye-') and any(k in s for k in ['photoreceptor','amacrine','bipolar','ganglion','horizontal','retinal','retina','lens','muller']): return 'EYE'
    if any(k in s for k in ['neuron','neural','glia','oligodendro','astrocyte','schwann','radial glia','neural tube','purkinje','granule','retinal progenitor']): return 'NEURAL'
    if any(k in s for k in ['erythro','megakaryo','myeloid','lymphoid','t cell','b cell','nk cell','hematop','thymocyte','antigen presenting','microglia']): return 'BLOOD_IMMUNE'
    if any(k in s for k in ['endothel','vascular','pericyte']): return 'ENDOTHELIAL'
    if any(k in s for k in ['hepato','pancrea','acinar','ductal','enteroendocrine','epitheli','lung','alveolar','intestin','gut','stomach','hepatic','islet','goblet','enterocyte','squamous']): return 'ENDO_EPITHELIAL'
    if any(k in s for k in ['mesenchym','stromal','fibroblast','chondro','osteo','myocyte','skeletal muscle','cardiac','cardiomyocyte','smooth muscle','satellite','connective','metanephric','mesoderm','limb','myofibroblast']): return 'MESODERM'
    return 'OTHER'

# ── curated ground-truth master regulators (conservative, unambiguous fate) ──
GROUND_TRUTH = {
    # neural (CNS/PNS neurons + glia)
    'NEUROG1':'NEURAL','NEUROG2':'NEURAL','NEUROD1':'NEURAL','NEUROD2':'NEURAL','ASCL1':'NEURAL',
    'OLIG2':'NEURAL','SOX1':'NEURAL','POU4F1':'NEURAL','NHLH1':'NEURAL','INSM1':'NEURAL','LHX2':'NEURAL',
    # eye / retina
    'CRX':'EYE','NRL':'EYE','NR2E3':'EYE','RAX':'EYE','VSX2':'EYE','ATOH7':'EYE',
    # mesoderm (paraxial/lateral/cardiac/muscle/skeletal)
    'MSGN1':'MESODERM','TBX6':'MESODERM','MESP1':'MESODERM','HAND1':'MESODERM','HAND2':'MESODERM',
    'FOXF1':'MESODERM','TWIST1':'MESODERM','PRRX1':'MESODERM','MEOX1':'MESODERM','MYOD1':'MESODERM',
    'MYF5':'MESODERM','MYOG':'MESODERM','NKX2-5':'MESODERM','TBX5':'MESODERM','RUNX2':'MESODERM','SOX9':'MESODERM',
    # endoderm / epithelial
    'FOXA2':'ENDO_EPITHELIAL','FOXA1':'ENDO_EPITHELIAL','HNF4A':'ENDO_EPITHELIAL','HNF1B':'ENDO_EPITHELIAL',
    'PDX1':'ENDO_EPITHELIAL','NKX2-1':'ENDO_EPITHELIAL','GATA6':'ENDO_EPITHELIAL','ONECUT1':'ENDO_EPITHELIAL','GRHL3':'ENDO_EPITHELIAL',
    # blood / endothelial
    'GATA1':'BLOOD_IMMUNE','TAL1':'BLOOD_IMMUNE','RUNX1':'BLOOD_IMMUNE','SPI1':'BLOOD_IMMUNE','KLF1':'BLOOD_IMMUNE',
    'LMO2':'BLOOD_IMMUNE','GATA2':'BLOOD_IMMUNE','ETV2':'ENDOTHELIAL','ERG':'ENDOTHELIAL','FLI1':'ENDOTHELIAL',
    # trophoblast / extraembryonic
    'TFAP2C':'TROPHOBLAST','GCM1':'TROPHOBLAST','ELF5':'TROPHOBLAST',
}


def main():
    cos = pd.read_csv(COS, index_col=0); cos.index = cos.index.astype(str).str.upper()
    ann = pd.read_csv(ANN).set_index('TF'); ann.index = ann.index.astype(str).str.upper()
    types = list(cos.columns)
    tlin = pd.Series({t: lineage_of(t) for t in types})
    print('lineage sizes (of 172 cell types):', tlin.value_counts().to_dict())
    unmapped = [t for t in types if tlin[t] == 'OTHER']
    print(f'unmapped (OTHER): {len(unmapped)} e.g. {unmapped[:6]}')

    gt = {k.upper(): v for k, v in GROUND_TRUTH.items()}
    present = [t for t in gt if t in cos.index]
    print(f'\nground-truth TFs: {len(gt)} curated, {len(present)} present in annotation (n>=50)')
    lineages = sorted(set(tlin) - {'OTHER'})

    rows = []
    for tf in present:
        row = cos.loc[tf]
        # lineage-aggregated prediction = mean cosine over each lineage's cell types
        lscore = {L: row[[t for t in types if tlin[t] == L]].mean() for L in lineages}
        pred_lin = max(lscore, key=lscore.get)
        topct = ann.loc[tf, 'top_celltype'] if tf in ann.index else row.idxmax()
        pred_topct_lin = lineage_of(str(topct))
        q = float(ann.loc[tf, 'fdr_qval']) if tf in ann.index and 'fdr_qval' in ann.columns else np.nan
        rows.append({'TF': tf, 'known': gt[tf], 'pred_lineage_agg': pred_lin,
                     'pred_topcelltype': topct, 'pred_topct_lineage': pred_topct_lin,
                     'q': q, 'correct_agg': pred_lin == gt[tf], 'correct_topct': pred_topct_lin == gt[tf]})
    df = pd.DataFrame(rows)
    df.to_csv(f'{HERE}/lineage_accuracy_per_tf.csv', index=False)

    n = len(df); chance = 1.0 / len(lineages)
    accA = df['correct_agg'].mean(); accB = df['correct_topct'].mean()
    conf = df[df['q'] < 0.10]
    print(f"\n=== LINEAGE PREDICTION ACCURACY (n={n} ground-truth TFs; {len(lineages)} lineages; chance={chance:.2f}) ===")
    print(f"  lineage-aggregated:      {accA:.2f}  ({df['correct_agg'].sum()}/{n})")
    print(f"  top-cell-type→lineage:   {accB:.2f}  ({df['correct_topct'].sum()}/{n})")
    print(f"  among FDR-confident (q<0.10, n={len(conf)}): agg {conf['correct_agg'].mean():.2f}" if len(conf) else "  (no q<0.10 TFs in GT)")
    print("\n  per-lineage recall (lineage-aggregated):")
    for L in lineages:
        sub = df[df['known'] == L]
        if len(sub): print(f"    {L:16s} {sub['correct_agg'].mean():.2f}  ({sub['correct_agg'].sum()}/{len(sub)})")
    print("\n  per-TF (lineage-aggregated):")
    print(df.sort_values('known')[['TF', 'known', 'pred_lineage_agg', 'q', 'correct_agg']].to_string(index=False))

    # confusion matrix figure (known × predicted, lineage-aggregated)
    cm = pd.crosstab(df['known'], df['pred_lineage_agg']).reindex(index=lineages, columns=lineages, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm.values, cmap='Blues')
    ax.set_xticks(range(len(lineages))); ax.set_xticklabels(lineages, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(lineages))); ax.set_yticklabels(lineages, fontsize=8)
    ax.set_xlabel('predicted lineage'); ax.set_ylabel('known lineage')
    for i in range(len(lineages)):
        for j in range(len(lineages)):
            v = cm.values[i, j]
            if v: ax.text(j, i, v, ha='center', va='center', color='white' if v > cm.values.max()/2 else 'black', fontsize=9)
    ax.set_title(f'TF→lineage: known vs predicted (shift-cosine)\nacc {accA:.2f} ({df.correct_agg.sum()}/{n}), chance {chance:.2f}', fontsize=11)
    fig.tight_layout(); p = f'{FIG}/exp17_lineage_confusion.png'; fig.savefig(p, dpi=140)
    json.dump({'n_gt_present': n, 'n_lineages': len(lineages), 'chance': chance,
               'acc_lineage_agg': float(accA), 'acc_topct': float(accB),
               'acc_conf_q010': float(conf['correct_agg'].mean()) if len(conf) else None,
               'n_conf_q010': int(len(conf))}, open(f'{HERE}/lineage_accuracy_stats.json', 'w'), indent=2)
    print(f'\nsaved → lineage_accuracy_per_tf.csv, lineage_accuracy_stats.json, {p}')


if __name__ == '__main__':
    main()
