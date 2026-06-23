"""No-compression accuracy: rank all 172 cell types by cosine and score recall@k against the
biologically-correct cell types for each ground-truth master TF. Nothing is bucketed/dropped.

Per TF: acceptable = the 172 types whose name matches the TF's known-fate keyword(s); rank the
172 by cosine; recall@k = a correct type is in top-k; best_rank = min rank of a correct type.

Usage: python recall_at_k.py <cosine_csv> [topcol_for_reference_label]
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# per-TF acceptable cell-type keyword patterns (substring, lowercased) over the 172 "Organ-CellType" names
TARGETS = {
    # neural (CNS/PNS neurons + glia)
    **{tf: ['neuron', 'neural', 'glia', 'oligodendro', 'astrocyte'] for tf in
       ['NEUROG1','NEUROG2','NEUROD1','NEUROD2','ASCL1','OLIG2','SOX1','POU4F1','NHLH1','INSM1','LHX2']},
    # eye / retina
    'CRX':['photoreceptor'], 'NRL':['photoreceptor'], 'NR2E3':['photoreceptor'],
    'RAX':['retinal','photoreceptor','muller'], 'VSX2':['bipolar','retinal','amacrine'], 'ATOH7':['ganglion','retinal'],
    # mesoderm / muscle / skeletal / cardiac
    **{tf: ['mesenchym','muscle','myocyte','cardiac','chondro','osteo','stromal','fibroblast','limb','satellite'] for tf in
       ['MSGN1','TBX6','MESP1','HAND1','HAND2','FOXF1','TWIST1','PRRX1','MEOX1','SOX9','RUNX2']},
    'MYOD1':['muscle','myocyte','satellite'], 'MYF5':['muscle','myocyte','satellite'], 'MYOG':['muscle','myocyte','satellite'],
    'NKX2-5':['cardiac','cardiomyocyte'], 'TBX5':['cardiac','cardiomyocyte'],
    # endoderm / epithelial
    **{tf: ['hepato','pancrea','lung','alveolar','intestin','gut','stomach','epitheli','acinar','ductal','enteroendocrine','islet'] for tf in
       ['FOXA2','FOXA1','HNF4A','HNF1B','PDX1','NKX2-1','GATA6','ONECUT1','GRHL3']},
    # blood / immune
    **{tf: ['erythro','myeloid','megakaryo','lymphoid','hematop','thymocyte','mast','blood'] for tf in
       ['GATA1','TAL1','RUNX1','SPI1','KLF1','LMO2','GATA2']},
    # endothelial
    'ETV2':['endothel','vascular'], 'ERG':['endothel','vascular'], 'FLI1':['endothel','vascular'],
    # trophoblast
    'TFAP2C':['trophoblast','syncytio','placenta'], 'GCM1':['trophoblast','syncytio','placenta'], 'ELF5':['trophoblast','syncytio','placenta'],
}


def main():
    cos = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else f'{HERE}/joung_human_cosine.csv', index_col=0)
    cos.index = cos.index.astype(str).str.upper()
    types = np.array([str(c) for c in cos.columns]); C = len(types); tl = np.array([t.lower() for t in types])
    gt = {k.upper(): v for k, v in TARGETS.items()}
    present = [t for t in gt if t in cos.index]
    r1 = r5 = r10 = 0; ranks = []; rows = []
    for tf in present:
        order = np.argsort(-cos.loc[tf].values)            # 172 type indices, best first
        acc = np.array([any(kw in tl[i] for kw in gt[tf]) for i in range(C)])
        if acc.sum() == 0: continue
        pos = np.where(acc[order])[0]                       # ranks (0-based) of acceptable types
        best = int(pos.min()) + 1                           # 1-based best rank
        ranks.append(best); r1 += best <= 1; r5 += best <= 5; r10 += best <= 10
        rows.append({'TF': tf, 'n_acceptable': int(acc.sum()), 'best_rank': best, 'top_type': types[order[0]]})
    n = len(rows)
    print(f"\nreference: {os.path.basename(sys.argv[1] if len(sys.argv)>1 else 'joung_human_cosine.csv')}  | {C} cell types, {n} ground-truth TFs")
    print(f"  recall@1  = {r1/n:.2f} ({r1}/{n})   [chance ~{np.mean([r['n_acceptable']/C for r in rows]):.2f}]")
    print(f"  recall@5  = {r5/n:.2f} ({r5}/{n})")
    print(f"  recall@10 = {r10/n:.2f} ({r10}/{n})")
    print(f"  median best-rank of a correct type = {int(np.median(ranks))} of {C}  (chance median ~{C//2})")
    print(pd.DataFrame(rows).sort_values('best_rank')[['TF', 'n_acceptable', 'best_rank', 'top_type']].to_string(index=False))


if __name__ == '__main__':
    main()
