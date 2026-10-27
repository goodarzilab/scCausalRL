"""Build the SENA gene->pathway membership mask for the vocab-v2 gene axis and save it.

Mirrors the SENA-δ selection (via utils.bio_priors.build_go_bp_mask):
  - GO:BP terms with <= max_term_size genes total
  - require >= min_genes_in_axis of the term's genes present in our vocab
  - dedup: drop any term sharing > max_overlap of genes with an already-kept (larger) term
Yields non-redundant, specific pathways. The saved mask is M + λ(1−M): 1.0 for annotated
gene-pathway pairs, λ (leak) off-annotation. Each pathway = one latent coordinate.

Output: pathway_mask_v2.pt = {mask [n_genes, n_pathways], genes, pathways, lambda}.
Run: python evaluation/build_pathway_mask.py --lam 0.1
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import anndata as ad
from utils.bio_priors import build_go_bp_mask

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2.h5ad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=0.1, help='λ leak off-annotation')
    ap.add_argument('--max_term', type=int, default=30)
    ap.add_argument('--min_genes', type=int, default=5)
    ap.add_argument('--max_overlap', type=float, default=0.5)
    ap.add_argument('--max_pathways', type=int, default=0, help='0 = uncapped')
    ap.add_argument('--out', default=os.path.join(VDIR, 'pathway_mask_v2.pt'))
    a = ap.parse_args()

    genes = [str(g) for g in ad.read_h5ad(PROC, backed='r').var_names]
    print(f"[mask] vocab-v2 genes: {len(genes)}", flush=True)
    M, terms = build_go_bp_mask(
        genes, max_term_size=a.max_term, min_genes_in_axis=a.min_genes,
        max_overlap=a.max_overlap, leak=a.lam,
        max_pathways=(a.max_pathways or None), verbose=True)
    M = np.asarray(M, dtype=np.float32)
    in_set = (M > a.lam)                                        # annotated pairs
    cov = float((in_set.max(1)).mean())
    mean_genes = float(in_set.sum(0).mean())
    print(f"[mask] {M.shape[1]} pathways | mean {mean_genes:.1f} genes/pathway | "
          f"gene coverage {100*cov:.1f}% | λ={a.lam}", flush=True)
    torch.save({'mask': torch.from_numpy(M), 'genes': genes, 'pathways': list(terms),
                'lambda': a.lam}, a.out)
    print(f"[mask] saved -> {a.out}", flush=True)


if __name__ == '__main__':
    main()
