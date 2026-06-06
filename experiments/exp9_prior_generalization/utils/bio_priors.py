import hashlib
import numpy as np
import pandas as pd


def split_collectri_edges(pairs, prior_train_frac=1.0, prior_split_seed=0,
                          prior_split_mode='tf'):
    """Deterministically split CollecTRI (src, tgt) gene-NAME pairs into
    train/test, for the prior-generalization ablation.

    The assignment is a pure function of the edge (or TF) identity via a stable
    md5 hash — NOT a shuffle of a list — so it is independent of input order,
    the gene set, n_hvg, or CollecTRI version. This guarantees the eval's
    held-out set is exactly the one training never supervised, even across runs.

    Args:
        pairs: iterable of (source_gene, target_gene) UPPERCASE name tuples.
        prior_train_frac: fraction assigned to TRAIN (1.0 ⇒ all train).
        prior_split_seed: salt for the hash.
        prior_split_mode: 'tf'   → split by TF name (a TF's edges are all train
                                    or all test; controls out-degree leakage), or
                          'edge' → split per (src,tgt) edge.

    Returns: (train_set, test_set), each a set of (src, tgt) tuples.
    """
    train, test = set(), set()
    if prior_train_frac >= 1.0:
        return set(pairs), set()
    for src, tgt in sorted(set(pairs)):
        key = src if prior_split_mode == 'tf' else f"{src}>{tgt}"
        h = hashlib.md5(f"{prior_split_seed}|{key}".encode()).hexdigest()
        bucket = int(h, 16) % 1000
        (train if bucket < prior_train_frac * 1000 else test).add((src, tgt))
    return train, test


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
