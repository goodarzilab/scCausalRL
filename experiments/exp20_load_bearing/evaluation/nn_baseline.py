"""Zero-shot RETRIEVAL baseline for held-out TFs (the baseline to beat): for each TEST effective TF,
find its nearest TRAIN TF by ESM2 protein-embedding cosine, and predict the test TF's residual as that
train TF's drift-removed residual. Score disc_resid + pearson (excl. the TF's own gene). This uses NO
training and NO test labels — a fair zero-shot floor. If the model < this baseline, it is leaving
signal on the table. Also reports the residual-space ORACLE NN (uses test residual to pick the
neighbour = optimistic upper bound) and the split-half data ceiling.

Run: python evaluation/nn_baseline.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.load_joung import load_joung

VDIR = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"
PROC = os.path.join(VDIR, "joung_singles_proc_v2b.h5ad")
EFF = os.path.join(VDIR, "effective_tf_genes.txt")
ESM = os.path.join(os.path.dirname(__file__), '..', 'data', 'joung_esm2_emb.npz')


def zrow(M):
    M = M - M.mean(1, keepdims=True); return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def disc(pred, obs):
    C = zrow(pred) @ zrow(obs).T; d = np.diag(C)[:, None]; T = len(pred)
    return float(((d > C).sum(1) / (T - 1)).mean())


def pear_noKO(pred, obs, genes, gidx):
    vals = []
    for k, g in enumerate(genes):
        m = np.ones(pred.shape[1], bool)
        if g in gidx: m[gidx[g]] = False
        a, b = pred[k, m], obs[k, m]
        if a.std() > 1e-9 and b.std() > 1e-9: vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals)) if vals else float('nan')


def main():
    dd = load_joung(proc_path=PROC, n_hvg=None, effective_tfs=EFF)
    de = dd['de_single']; gn = list(map(str, dd['gene_names'])); gidx = {g: i for i, g in enumerate(gn)}
    tr = [g for g in dd['pert_split']['train'] if g in de]
    te = [g for g in dd['pert_split']['test'] if g in de]
    drift = np.stack([np.asarray(de[g], np.float32) for g in tr]).mean(0)
    Rtr = np.stack([np.asarray(de[g], np.float32) for g in tr]) - drift
    Rte = np.stack([np.asarray(de[g], np.float32) for g in te]) - drift
    Ote = np.stack([np.asarray(de[g], np.float32) for g in te])   # raw obs (for scoring in disc_resid space)

    z = np.load(ESM, allow_pickle=True)
    esm = {str(g): e for g, e in zip(z['genes'], z['emb'])}
    have = lambda g: g in esm and np.linalg.norm(esm[g]) > 0
    print(f"=== NN baselines on v2b effective TFs === train={len(tr)} test={len(te)} | "
          f"ESM cover train={sum(map(have,tr))}/{len(tr)} test={sum(map(have,te))}/{len(te)}", flush=True)

    # ESM nearest-train (realistic zero-shot: pick neighbour by protein embedding only)
    Etr = np.stack([esm.get(g, np.zeros(1280)) for g in tr]); Ete = np.stack([esm.get(g, np.zeros(1280)) for g in te])
    Etr = Etr / (np.linalg.norm(Etr, axis=1, keepdims=True) + 1e-9); Ete = Ete / (np.linalg.norm(Ete, axis=1, keepdims=True) + 1e-9)
    nn_esm = (Ete @ Etr.T).argmax(1)
    pred_esm = Rtr[nn_esm]
    # residual-space oracle (uses test residual to pick neighbour — optimistic)
    nn_or = (zrow(Rte) @ zrow(Rtr).T).argmax(1); pred_or = Rtr[nn_or]
    # mean-train (predict the mean train residual for everything → disc chance by construction)
    pred_mean = np.repeat(Rtr.mean(0, keepdims=True), len(te), 0)

    print(f"\n  {'baseline':<28}{'disc_resid':>11}{'pear_noKO':>11}", flush=True)
    for nm, P in [('ESM nearest-train (REALISTIC)', pred_esm),
                  ('residual-NN oracle (upper bnd)', pred_or),
                  ('mean-train residual', pred_mean)]:
        print(f"  {nm:<28}{disc(P, Rte):>11.3f}{pear_noKO(P + drift, Ote, te, gidx):>11.3f}", flush=True)
    print(f"\n  reference: our best model (v2b_masked_go_l3) test disc_resid = 0.597", flush=True)
    print(f"  reference: test-set split-half data ceiling ~0.699", flush=True)


if __name__ == '__main__':
    main()
