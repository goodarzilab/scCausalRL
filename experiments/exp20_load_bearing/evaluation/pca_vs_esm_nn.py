"""Zero-shot descriptor comparison: is the control-co-expression PCA descriptor (Uhler/SCCVAE style)
zero-shot, and how does it compare to ESM2 for held-out-TF discrimination? Both are PER-GENE
descriptors (exist for unseen genes) → both are zero-shot capable. Hypothesis (from the code comment):
PCA co-expression is INFO-FREE for TFs silent in control, so ESM wins especially on silent TFs.

For each descriptor, run nearest-TRAIN-neighbour (by descriptor cosine) → predict test TF residual =
neighbour's residual → disc_resid over the held-out test TFs. Also split test TFs into SILENT vs
EXPRESSED (by the TF gene's own mean control expression) and report disc per subset.

Run: python evaluation/pca_vs_esm_nn.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.load_joung import load_joung
from utils.pert_descriptors import _control_pca, _regulon_pca

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad"); EFF = os.path.join(VDIR, "effective_tf_genes.txt")
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')


def zrow(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc_subset(pred, obs, mask):
    # disc over the FULL test set ranking, then average only over the `mask` query TFs
    C = zrow(pred) @ zrow(obs).T; d = np.diag(C)[:, None]; T = len(pred)
    per = (d > C).sum(1) / (T - 1)
    return float(per[mask].mean())


def nn_pred(Etr, Ete, Rtr):
    Etr = Etr / (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9)   # per-row L2 → cosine NN
    Ete = Ete / (np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-9)
    return Rtr[(Ete @ Etr.T).argmax(1)]


def main():
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]; te = [g for g in dd['pert_split']['test'] if g in de]
    import scipy.sparse as sp
    X = dd['adata'].X; X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    ctrl = np.array(dd['control_idx'])
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    Rtr = np.stack([np.asarray(de[g], np.float32) for g in tr]) - drift
    Rte = np.stack([np.asarray(de[g], np.float32) for g in te]) - drift

    tri = [gidx[g] for g in tr]; tei = [gidx[g] for g in te]
    # descriptors (all per-gene → indexable for train/test TFs)
    Dpca = _control_pca(X, ctrl, 50, 2000, 0)                                  # [G,50] control co-expr PCA
    Dreg = _regulon_pca(gn, 50)                                                 # [G,50] CollecTRI regulon PCA
    z = np.load(ESM, allow_pickle=True); esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}
    dim = len(next(iter(esm.values()))); Desm = np.stack([esm.get(g, np.zeros(dim)) for g in gn])

    # silent vs expressed test TFs, by the TF gene's OWN mean control expression
    ctrl_expr = X[ctrl][:, tei].mean(0)                                         # [n_te]
    med = np.median(X[ctrl].mean(0))                                           # global gene-expression median
    silent = ctrl_expr < med; expr = ~silent
    print(f"=== PCA vs ESM zero-shot NN (v2b, {len(tr)} train / {len(te)} test TFs) ===", flush=True)
    print(f"  test TFs silent-in-control: {int(silent.sum())}/{len(te)} (below global median expr)", flush=True)
    print(f"\n  {'descriptor':<26}{'disc_all':>9}{'disc_silent':>12}{'disc_expr':>10}", flush=True)
    for nm, D in [('control-PCA (Uhler)', Dpca), ('regulon-PCA (CollecTRI)', Dreg), ('ESM2 protein', Desm)]:
        pred = nn_pred(D[tri], D[tei], Rtr)
        da = disc_subset(pred, Rte, np.ones(len(te), bool))
        ds = disc_subset(pred, Rte, silent); dx = disc_subset(pred, Rte, expr)
        print(f"  {nm:<26}{da:>9.3f}{ds:>12.3f}{dx:>10.3f}", flush=True)
    print(f"\n  (chance=0.50; both PCA and ESM are per-gene → both zero-shot. Q: does PCA collapse on silent TFs?)", flush=True)


if __name__ == '__main__':
    main()
