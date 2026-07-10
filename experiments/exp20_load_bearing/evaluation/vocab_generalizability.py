"""Parts 1b + 3: is zero-shot generalization to held-out effective TFs even possible on vocab-v2b,
and does the vocab still carry TF-specific (beyond-shared-drift) signal?

On the v2b effective-TF set (446 train / 49 test, seed 42 gene split):
 (3) TF-specificity: mean pairwise cosine of raw per-TF Δ (shared-drift dominance) and of drift-removed
     residuals; fraction of residual energy ORTHOGONAL to the shared drift = the "TF-specific %".
 (1b) generalizability: for each TEST TF residual, (a) max cosine to any TRAIN TF residual, (b) fraction
     of its energy captured by span(train residuals) [orthogonal => unpredictable], (c) a NEAREST-TRAIN
     baseline — predict each test TF's residual as its nearest train TF's residual (by residual cosine =
     oracle upper bound), score disc_resid; chance 0.5. If even the oracle-NN baseline ~0.5, test TFs are
     orthogonal to train and no method generalizes; if >>0.5, generalization is possible and the model
     is leaving signal on the table.

Run: python evaluation/vocab_generalizability.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad")
EFF = os.path.join(VDIR, "effective_tf_genes.txt")


def zrow(M):
    M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc(pred, truth):
    """fraction of other-TF truths that pred[i] beats truth[i] on (cosine). chance 0.5."""
    Zp, Zt = zrow(pred), zrow(truth); C = Zp @ Zt.T
    d = np.diag(C)[:, None]; T = pred.shape[0]
    return float(((d > C).sum(1) / (T - 1)).mean())


def main():
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; sp_map = dd['single_perts']
    tr = [g for g in dd['pert_split']['train'] if g in de]
    te = [g for g in dd['pert_split']['test'] if g in de]
    print(f"=== vocab-v2b generalizability + TF-specificity === train={len(tr)} test={len(te)} TFs", flush=True)

    raw_tr = np.stack([np.asarray(de[g], np.float32) for g in tr])
    raw_te = np.stack([np.asarray(de[g], np.float32) for g in te])
    drift = raw_tr.mean(0)                                   # shared program (train-only, leak-free)
    Rtr = raw_tr - drift; Rte = raw_te - drift               # drift-removed residuals

    # (3) TF-specificity of the vocab
    def mean_offdiag_cos(M):
        Z = zrow(M); C = Z @ Z.T; n = len(M)
        return float((C.sum() - np.trace(C)) / (n * (n - 1)))
    u = drift / (np.linalg.norm(drift) + 1e-9)
    along = np.abs(Rtr @ u)                                   # residual energy along shared dir
    tot = np.linalg.norm(Rtr, axis=1)
    tf_specific_frac = float(np.mean(1 - (along / (tot + 1e-9))**2))  # fraction of residual var ⟂ shared
    print("\n(3) TF-SPECIFICITY on vocab-v2b:", flush=True)
    print(f"  mean pairwise cosine of RAW Δ (shared-drift dominance): {mean_offdiag_cos(raw_tr):.3f}", flush=True)
    print(f"  mean pairwise cosine of RESIDUALS (TF-to-TF):           {mean_offdiag_cos(Rtr):+.3f}", flush=True)
    print(f"  residual magnitude / raw magnitude:                     {tot.mean()/np.linalg.norm(raw_tr,axis=1).mean():.2%}", flush=True)
    print(f"  residual energy ORTHOGONAL to shared drift (TF-specific %): {100*tf_specific_frac:.1f}%", flush=True)

    # (1b) generalizability of the test set from train
    Qtr, _ = np.linalg.qr(Rtr.T)                              # orthonormal basis of train-residual span [G,k]
    cap = ((Rte @ Qtr)**2).sum(1) / ((Rte**2).sum(1) + 1e-9)  # fraction of each test residual in train span
    Ztr, Zte = zrow(Rtr), zrow(Rte)
    nn_cos = (Zte @ Ztr.T).max(1)                            # nearest-train residual cosine per test TF
    nn_idx = (Zte @ Ztr.T).argmax(1)
    nn_pred = Rtr[nn_idx]                                    # oracle nearest-train prediction
    print("\n(1b) GENERALIZABILITY of the 49 test TFs from the 446 train TFs:", flush=True)
    print(f"  test residual captured by span(train residuals): mean {100*cap.mean():.1f}% "
          f"(train has {Rtr.shape[0]} dims for a {Rtr.shape[1]}-gene space -> near-full span; expect high)", flush=True)
    print(f"  nearest-TRAIN residual cosine per test TF: mean {nn_cos.mean():.3f} median {np.median(nn_cos):.3f} "
          f"(>0 => a similar train TF exists)", flush=True)
    print(f"  ORACLE nearest-train baseline disc_resid: {disc(nn_pred, Rte):.3f}  (chance 0.5)", flush=True)
    print(f"    -> if ~0.5: test TFs are orthogonal to train, NO method generalizes.", flush=True)
    print(f"    -> if >>0.5: generalization IS possible; the model (test disc ~0.58) leaves signal unused.", flush=True)
    # for calibration: split-half reliability ceiling on the test TFs themselves
    rng = np.random.default_rng(0); rA, rB = [], []
    for g in te:
        idx = rng.permutation(np.asarray(sp_map[g])); h = len(idx)//2
        if h < 25: continue
        X = dd['adata'].X; ctrl = np.asarray(dd['control_idx'])
        cm = (X[ctrl].toarray() if sp.issparse(X) else np.asarray(X)[ctrl]).mean(0)
        rA.append((np.asarray((X[idx[:h]].toarray() if sp.issparse(X) else X[idx[:h]]).mean(0)) - cm) - drift)
        rB.append((np.asarray((X[idx[h:2*h]].toarray() if sp.issparse(X) else X[idx[h:2*h]]).mean(0)) - cm) - drift)
    if len(rA) >= 5:
        print(f"  test-set data ceiling (split-half disc on {len(rA)} test TFs): {disc(np.stack(rA), np.stack(rB)):.3f}", flush=True)


if __name__ == '__main__':
    main()
