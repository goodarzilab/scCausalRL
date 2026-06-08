import numpy as np
import pandas as pd


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
