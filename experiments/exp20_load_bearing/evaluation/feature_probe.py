"""GATE probe (D3 generalized): which per-TF INPUT FEATURE carries generalizable TF-identity?

Model-free. For each candidate feature F[g] (a per-gene vector), fit Ridge(F[train] →
observed Δ[train]) over the train-split TFs and measure retrieval on HELD-OUT test TFs
(does the predicted Δ retrieve its own observed Δ among the test set?). A feature whose
held-out retrieval clears chance carries TF-identity that a shift encoder could exploit;
one at chance is an input bottleneck no encoder can fix. This decides what to retrain with
BEFORE retraining anything.

Candidates: control-coexpression PCA (baseline), CollecTRI targets (raw + PCA), ESM2 protein
embedding, TF DBD family (Lambert 2018), and concatenations. Reports train (in-sample) and
test (held-out) retrieval_top1 + discrimination, over all test TFs and over feature-covered
test TFs. chance top1 = 1/N_test, disc ≈ 0.5.

Run (high-mem node):  python evaluation/feature_probe.py --tag epoch80
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data')


def zrows(M):
    M = np.asarray(M, np.float64); M = M - M.mean(1, keepdims=True)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def retrieval(pred, obs):
    Pz, Oz = zrows(pred), zrows(obs); C = Pz @ Oz.T; T = C.shape[0]
    top1 = float((C.argmax(1) == np.arange(T)).mean())
    diag = np.diag(C)[:, None]; disc = float((((diag > C).sum(1)) / (T - 1)).mean())
    return top1, disc


def pca_reduce(M, q):
    import torch
    Mt = torch.as_tensor(M, dtype=torch.float32); qq = min(q, *Mt.shape)
    _, _, V = torch.pca_lowrank(Mt, q=qq, center=True)
    return ((Mt - Mt.mean(0, keepdim=True)) @ V).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='epoch80')
    ap.add_argument('--alpha', type=float, default=50.0)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'feature_probe'); os.makedirs(out, exist_ok=True)
    import torch, scipy.sparse as sp
    from data.load_joung import load_joung
    from utils.bio_priors import load_collectri
    from utils.pert_descriptors import _control_pca

    dd = load_joung(n_hvg=None); de = dd['de_single']
    gene_names = dd['gene_names']; gidx = {g: i for i, g in enumerate(gene_names)}; G = len(gene_names)
    Xfull = dd['adata'].X; Xfull = Xfull.toarray() if sp.issparse(Xfull) else np.asarray(Xfull)
    train = [g for g in dd['pert_split']['train'] if g in de]
    test = [g for g in dd['pert_split']['test'] if g in de]
    O_tr = np.stack([np.asarray(de[g], np.float32) for g in train])
    O_te = np.stack([np.asarray(de[g], np.float32) for g in test])
    print(f"[probe] train={len(train)} test={len(test)} genes={G}", flush=True)

    # ---- build candidate per-gene feature matrices [G, dim] ----
    feats = {}
    # control-coexpression PCA (the current baseline) — recomputed from control cells
    feats['ctrl_pca50'] = _control_pca(Xfull, dd['control_idx'], 50, 2000, 42)  # [G, 50]
    # CollecTRI targets (g as TF → its targets)
    net = load_collectri(organism='human'); R = np.zeros((G, G), np.float32)
    for s, t in zip(net['source'].astype(str).str.upper(), net['target'].astype(str).str.upper()):
        si, ti = gidx.get(s), gidx.get(t)
        if si is not None and ti is not None: R[si, ti] = 1.0
    feats['regulon_raw'] = R
    feats['regulon_pca100'] = pca_reduce(R, 100)
    # ESM2 protein embedding
    z = np.load(os.path.join(DATA, 'joung_esm2_emb.npz'), allow_pickle=True)
    emb = {str(g): v for g, v in zip(z['genes'], z['emb'])}; ed = len(next(iter(emb.values())))
    E = np.stack([emb.get(g, np.zeros(ed, np.float32)) for g in gene_names]).astype(np.float32)
    feats['esm_raw'] = E
    feats['esm_pca100'] = pca_reduce(E, 100)
    # TF DBD family (Lambert 2018) one-hot
    import pandas as pd
    lam = pd.read_csv(os.path.join(DATA, 'lambert2018_tfs.csv'))
    fam = {str(s).upper(): str(d) for s, d in zip(lam['HGNC symbol'], lam['DBD']) if str(d) != 'nan'}
    families = sorted(set(fam.values())); fi = {f: i for i, f in enumerate(families)}
    Fam = np.zeros((G, len(families)), np.float32)
    for i, g in enumerate(gene_names):
        if g.upper() in fam: Fam[i, fi[fam[g.upper()]]] = 1.0
    feats['tf_family'] = Fam
    # concatenations (z-scored blocks)
    def zc(*names):
        return np.concatenate([(feats[n] - feats[n].mean(0)) / (feats[n].std(0) + 1e-6) for n in names], 1)
    feats['esm+regulon_pca'] = zc('esm_pca100', 'regulon_pca100')
    feats['esm+family'] = zc('esm_pca100', 'tf_family')
    feats['esm+regulon+family'] = zc('esm_pca100', 'regulon_pca100', 'tf_family')
    feats['ctrl+esm+regulon'] = zc('ctrl_pca50', 'esm_pca100', 'regulon_pca100')

    # drift-removed residual targets: subtract the shared across-TF mean Δ (estimated on
    # train) so the probe tests prediction of the TF-SPECIFIC component, not the shared
    # differentiation drift (which dominates raw Δ and masks retrieval at chance).
    drift = O_tr.mean(0, keepdims=True)
    O_tr_res, O_te_res = O_tr - drift, O_te - drift

    from sklearn.linear_model import Ridge
    tr_i = [gidx[g] for g in train]; te_i = [gidx[g] for g in test]
    rows = []
    for name, F in feats.items():
        Xtr, Xte = F[tr_i], F[te_i]
        cov_te = float((np.abs(Xte).sum(1) > 0).mean())                      # feature coverage on test
        rg = Ridge(alpha=args.alpha).fit(Xtr, O_tr)
        tr_top1, tr_disc = retrieval(rg.predict(Xtr), O_tr)
        te_top1, te_disc = retrieval(rg.predict(Xte), O_te)
        # decisive: fit + retrieve on the drift-removed residual (TF-specific component)
        rgr = Ridge(alpha=args.alpha).fit(Xtr, O_tr_res)
        res_top1, res_disc = retrieval(rgr.predict(Xte), O_te_res)
        # covered-only test retrieval (optimistic: TFs that actually have the feature)
        cmask = np.abs(Xte).sum(1) > 0
        if cmask.sum() >= 3:
            ct1, cd = retrieval(rg.predict(Xte[cmask]), O_te[cmask])
        else:
            ct1, cd = np.nan, np.nan
        rows.append(dict(feature=name, dim=F.shape[1], cov_test=round(cov_te, 3),
                         train_top1=round(tr_top1, 4), train_disc=round(tr_disc, 4),
                         test_top1=round(te_top1, 4), test_disc=round(te_disc, 4),
                         test_top1_resid=round(res_top1, 4), test_disc_resid=round(res_disc, 4),
                         test_top1_covered=round(ct1, 4)))
        print(f"  {name:22s} dim={F.shape[1]:5d} cov={cov_te:.2f} | "
              f"train_top1={tr_top1:.3f} | TEST_top1={te_top1:.3f} disc={te_disc:.3f} | "
              f"RESID test_top1={res_top1:.3f} disc={res_disc:.3f}", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows).sort_values('test_top1', ascending=False)
    df.to_csv(os.path.join(out, f'{args.tag}_probe.csv'), index=False)
    json.dump({'tag': args.tag, 'n_test': len(test), 'chance_top1': 1.0 / len(test), 'rows': rows},
              open(os.path.join(out, f'{args.tag}_probe.json'), 'w'), indent=2)
    print(f"\nchance top1 = {1.0/len(test):.4f}; disc chance = 0.5")
    print(df.to_string(index=False))
    print(f"saved → {out}/{args.tag}_probe.csv")


if __name__ == '__main__':
    main()
