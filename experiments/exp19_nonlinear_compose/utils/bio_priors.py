import os
import pickle
import numpy as np
import pandas as pd

# Cache of the GO:BP gene-set library (Enrichr GO_Biological_Process_2023), pulled
# once via gseapy. This is the SAME ground-truth source SENA-discrepancy-VAE (de la
# Fuente et al., ICLR 2025) uses for its pathway mask — GO Biological Process gene
# sets, NOT a regulatory (TF→target) prior. It anchors latent programs to named
# processes *structurally in the encoder*; it is orthogonal to CollecTRI (which we
# keep eval-only), so there is no prior-memorization circularity.
_GO_BP_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', 'genesets', 'go_bp_2023.pkl')


def load_go_bp_genesets():
    """Return dict {term_name: [GENE,...]} of GO:BP gene sets (upper-cased).

    Reads the local cache if present, else fetches via gseapy (Enrichr
    GO_Biological_Process_2023) and caches it.
    """
    if os.path.exists(_GO_BP_CACHE):
        lib = pickle.load(open(_GO_BP_CACHE, 'rb'))
    else:
        import gseapy
        lib = gseapy.get_library(name='GO_Biological_Process_2023', organism='Human')
        os.makedirs(os.path.dirname(_GO_BP_CACHE), exist_ok=True)
        pickle.dump(lib, open(_GO_BP_CACHE, 'wb'))
    return {k: [str(g).upper() for g in v] for k, v in lib.items()}


def build_go_bp_mask(gene_names, max_term_size=30, min_genes_in_axis=5,
                     max_overlap=0.5, leak=0.1, max_pathways=None, verbose=True):
    """Build the SENA gene→pathway membership mask M for our gene axis.

    Mirrors SENA-discrepancy-VAE §4.2 selection: GO:BP terms with ≤ max_term_size
    genes, ≥ min_genes_in_axis present in our axis, greedily de-duplicated so no kept
    term overlaps a previously-kept one by > max_overlap (overlap = |A∩B|/min(|A|,|B|)),
    largest-first. (Ancestor removal is approximated by this overlap filter.)

    M[i, k] = 1.0   if gene i ∈ pathway k
            = leak  otherwise     (the tunable λ leak: lets data add off-pathway
                                    contributions; λ→0 = max interpretability.)

    Returns: (M float32 ndarray [n_genes, K], kept_terms list[str]).
    Genes in no kept pathway get an all-`leak` row (still weakly trainable).
    """
    lib = load_go_bp_genesets()
    g2i = {str(g).upper(): i for i, g in enumerate(gene_names)}
    n = len(gene_names)

    # candidate terms: small, well-represented in our axis
    cands = []
    for term, genes in lib.items():
        if len(genes) > max_term_size:
            continue
        idx = sorted({g2i[g] for g in genes if g in g2i})
        if len(idx) >= min_genes_in_axis:
            cands.append((term, set(idx)))
    cands.sort(key=lambda t: -len(t[1]))                      # largest-first

    kept = []
    for term, idx in cands:
        dup = False
        for _, kidx in kept:
            inter = len(idx & kidx)
            if inter and inter / min(len(idx), len(kidx)) > max_overlap:
                dup = True
                break
        if not dup:
            kept.append((term, idx))
        if max_pathways and len(kept) >= max_pathways:
            break

    K = len(kept)
    M = np.full((n, K), float(leak), dtype=np.float32)
    for k, (_, idx) in enumerate(kept):
        for i in idx:
            M[i, k] = 1.0
    covered = int((M > leak).any(axis=1).sum())
    if verbose:
        print(f"[SENA mask] GO:BP terms: {len(lib)} → candidates {len(cands)} → kept {K} "
              f"(≤{max_term_size} genes, ≥{min_genes_in_axis} in-axis, overlap≤{max_overlap}); "
              f"leak={leak}; genes covered {covered}/{n} ({100*covered/n:.1f}%)")
    return M, [t for t, _ in kept]


def load_collectri(organism='human'):
    """
    Load CollecTRI TF-target network using decoupler.
    Returns pandas DataFrame with columns 'source' and 'target'.
    Falls back to empty DataFrame if not available.

    Uses dc.op.collectri(organism=organism) API.
    """
    try:
        import decoupler as dc
        net = dc.op.collectri(organism=organism)
        if 'source' in net.columns and 'target' in net.columns:
            net = net[['source', 'target']].copy()
            # Normalise to uppercase for gene-name matching
            net['source'] = net['source'].astype(str).str.upper()
            net['target'] = net['target'].astype(str).str.upper()
            net = net.drop_duplicates()
            print(f"CollecTRI: loaded {len(net)} TF→target pairs for organism='{organism}'")
            return net
        else:
            print("CollecTRI: unexpected column names:", net.columns.tolist())
            return pd.DataFrame(columns=['source', 'target'])
    except ImportError:
        print("decoupler not installed. CollecTRI prior will be skipped.")
        return pd.DataFrame(columns=['source', 'target'])
    except Exception as e:
        print(f"Failed to load CollecTRI: {e}")
        return pd.DataFrame(columns=['source', 'target'])


def get_tf_target_pairs(collectri_net, gene_names):
    """
    Get indices of TF-target pairs that exist in our gene set.
    
    Returns: list of (tf_idx, target_idx) tuples
    """
    # Upper-case lookup to handle mouse/human gene name capitalisation differences
    gene_to_idx_upper = {g.upper(): i for i, g in enumerate(gene_names)}

    pairs = []
    for _, row in collectri_net.iterrows():
        tf = str(row['source']).upper()
        target = str(row['target']).upper()
        tf_idx = gene_to_idx_upper.get(tf)
        tgt_idx = gene_to_idx_upper.get(target)
        if tf_idx is not None and tgt_idx is not None:
            pairs.append((tf_idx, tgt_idx))
    return pairs


def compute_grn_auprc(G_feat_abs, collectri_pairs, gene_names, n_genes):
    """
    Compute AUPRC of learned GRN against CollecTRI ground truth.
    
    G_feat_abs: array [n_genes, n_genes] — absolute values of G_feat
    collectri_pairs: list of (tf_idx, target_idx)
    """
    from sklearn.metrics import average_precision_score
    
    # Build binary label matrix
    y_true = np.zeros(n_genes * n_genes)
    y_score = G_feat_abs.ravel()
    
    # Set positives
    for tf_idx, tgt_idx in collectri_pairs:
        if tf_idx < n_genes and tgt_idx < n_genes:
            y_true[tf_idx * n_genes + tgt_idx] = 1
    
    if y_true.sum() == 0:
        return 0.0
    
    try:
        return average_precision_score(y_true, y_score)
    except Exception as e:
        print(f"AUPRC computation failed: {e}")
        return 0.0
