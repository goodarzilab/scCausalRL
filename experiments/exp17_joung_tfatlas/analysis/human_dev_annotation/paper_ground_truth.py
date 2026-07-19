"""Authoritative TF->cell-type ground truth from Joung et al. 2023 (Cell 186:209), the paper
that GENERATED the TF Atlas. They annotated TF-induced profiles against the SAME Cao 2020 human
fetal atlas we use (their ref 30 = GSE156793), then EXPERIMENTALLY VALIDATED candidate TFs
(Fig 3D nominations -> Fig 4 marker-gene + immunostaining validation, 22/24 confirmed).

This replaces the hand-curated 38-TF set (which spanned fates a 7-day hESC differentiation can't
reach, e.g. photoreceptor/blood). Two products:
  GROUND_TRUTH  : validated TF -> paper cell type, with Cao-172 column keyword patterns
  REACHABLE     : the ~23 cell types this assay actually produces (Table S4B) -> Cao keywords,
                  used to restrict the candidate set so methods aren't penalized for ranking
                  unreachable types (which no method could ever get right here).
Run: python paper_ground_truth.py   (writes joung_paper_ground_truth.csv)
"""
import os, csv

# ── EXPERIMENTALLY VALIDATED TF -> cell type (Joung Fig 3D nominated, Fig 4 validated) ──
# value = (paper cell-type label, [keyword substrings matched against Cao-172 column names])
GROUND_TRUTH = {
    'NEUROD1': ('neuron (PNS)',        ['neuron']),
    'FERD3L':  ('neuron (PNS)',        ['neuron']),
    'LMX1B':   ('neuron (PNS)',        ['neuron']),
    'T':       ('smooth muscle',       ['smooth muscle']),   # TBXT / Brachyury (RefSeq name 'T')
    'MSGN1':   ('smooth muscle',       ['smooth muscle']),
    'ZBTB7B':  ('smooth muscle',       ['smooth muscle']),
    'DMRTA2':  ('smooth muscle',       ['smooth muscle']),
    'BHLHA9':  ('smooth muscle',       ['smooth muscle']),
    'PTF1A':   ('EMT smooth muscle',   ['smooth muscle']),
    'FLI1':    ('vascular endothelial',['endothel', 'vascular']),
    'CDX1':    ('metanephric',         ['metanephric']),
    'CDX2':    ('metanephric',         ['metanephric']),
    'KLF4':    ('intestinal epithelial',['intestinal epithel']),
    'HNF4A':   ('intestinal epithelial',['intestinal epithel']),
    'NR5A2':   ('intestinal epithelial',['intestinal epithel']),
    'NHLH1':   ('lung ciliated epithelial',['ciliated epithel', 'bronchiolar', 'alveolar']),
    'ASCL2':   ('lung ciliated epithelial',['ciliated epithel', 'bronchiolar', 'alveolar']),
    'GRHL3':   ('trophoblast',         ['trophoblast', 'syncytio', 'cytotrophoblast']),
}

# ── reachable cell-type vocabulary (Table S4B: the 23 types the assay produces) -> Cao keywords ──
# restricts the ranking pool so chance is computed over types that are actually attainable here.
REACHABLE_KEYWORDS = [
    'neuron', 'neural', 'glia',                                   # neuroectoderm / neuron / ENS
    'endothel', 'vascular',                                       # vascular endothelial
    'smooth muscle', 'skeletal muscle', 'satellite', 'myocyte',   # mesoderm / muscle
    'cardio', 'cardiac',                                          # EOMES->iCM (Fig 5)
    'metanephric', 'ureteric',                                    # kidney
    'epithel', 'goblet', 'bronchiolar', 'alveolar',               # endoderm/ectoderm epithelia
    'stromal', 'mesenchym', 'mesothel',                           # stromal
    'trophoblast', 'syncytio', 'cytotrophoblast',                 # extraembryonic
    'hepatoblast',                                                # endoderm
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, 'joung_paper_ground_truth.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['TF', 'paper_cell_type', 'cao_keywords'])
        for tf, (ct, kws) in GROUND_TRUTH.items():
            w.writerow([tf, ct, '|'.join(kws)])
    print(f'wrote {len(GROUND_TRUTH)} validated TF->cell-type pairs -> {p}')
    print(f'cell types covered: {sorted(set(ct for ct, _ in GROUND_TRUTH.values()))}')


if __name__ == '__main__':
    main()
