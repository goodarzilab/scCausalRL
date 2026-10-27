"""(a) Joung-combo NON-ADDITIVITY ceiling — how additive are Joung combos observationally? Sets the max
   any interaction head could buy in-domain.  nonadd = ||Δ_combo - ΣΔ_single|| / ||Δ_combo||; add_corr =
   corr(Δ_combo, ΣΔ_single). If add_corr~1, combos are ~additive -> no head helps.

(b) Briggs-supervised combo-Δ READOUT — does a Ridge map z_post(combo) -> observed Briggs combo Δ predict
   HELD-OUT Briggs combos (vs the unsupervised do-op's -0.10)? This is the domain bridge (the proven route:
   fate readout hit +0.71). Report held-out per-pair corr, random-split + leave-TF-out CV, vs do-op baseline.
"""
import os, sys, argparse
import numpy as np, torch
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.setup_stack import add_stack_args, build_stack

EXP20 = "/home/sgoudarzi/scCausalRL/experiments/exp20_load_bearing"


def cc(a, b):
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--pool', default='curated')
    ap.add_argument('--mc5', default=f'{EXP20}/results/briggs/briggs_combo_delta_mc5.npz')
    ap.add_argument('--alpha', type=float, default=10.0)
    ap.add_argument('--save_readout', default=None, help='save the held-out-combo-trained direction readout (FateReward npz format) for MPC')
    ap.add_argument('--save_feats', default=None, help='save the observed-combo latent manifold (Z,combo) for the Mahalanobis support penalty')
    a = ap.parse_args()
    dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8); data.set_pool(a.pool)
    gene_names = list(map(str, data.gene_names)); g2i = data.gene_to_idx
    A = op.mean_A(); eps0 = data.eps_ctrl_mean.to(dev)
    ctrl_recon = op.decode(data.z0.unsqueeze(0))[0].detach().cpu().numpy()

    # ---------- (a) Joung-combo non-additivity (observational) ----------
    print("===== (a) JOUNG-COMBO NON-ADDITIVITY (head ceiling) =====", flush=True)
    nonadd, addcorr = [], []
    for key, de in data.de_combo.items():
        tfs = [t for t in key if t in data.de_single]
        if len(tfs) < 2:
            continue
        oc = np.asarray(de['lfc'], np.float32)
        ss = np.sum([np.asarray(data.de_single[t]['lfc'], np.float32) for t in tfs], 0)
        nonadd.append(float(np.linalg.norm(oc - ss) / (np.linalg.norm(oc) + 1e-9)))
        addcorr.append(cc(oc, ss))
    print(f"  n={len(nonadd)} Joung pairs | non-additive frac ||Δc-ΣΔs||/||Δc|| median={np.median(nonadd):.3f} "
          f"| corr(Δcombo, ΣΔsingle) median={np.nanmedian(addcorr):.3f}", flush=True)
    print(f"  => {'combos ~ADDITIVE (little head can buy)' if np.nanmedian(addcorr) > 0.85 else 'real non-additive residual (head has headroom)'}", flush=True)

    # ---------- (b) Briggs-supervised combo-Δ readout ----------
    print("\n===== (b) BRIGGS-SUPERVISED combo-Δ readout (z_post -> observed Briggs combo Δ) =====", flush=True)
    bz = np.load(a.mc5, allow_pickle=True)
    sidx = np.asarray(bz['shared_idx'])
    pairs = [[str(t) for t in pr if str(t) in g2i] for pr in bz['pairs']]
    keep = [i for i, p in enumerate(pairs) if len(p) >= 2]
    Y = np.stack([np.asarray(bz['obs_delta'][i], np.float32)[sidx] for i in keep])       # [n, |shared|] Briggs Δ
    setlist = [pairs[i] for i in keep]

    @torch.no_grad()
    def zpost(p):
        gi = [g2i[t] for t in p]
        return op.apply_sets(eps0, [gi], A)[0].cpu().numpy()
    Z = np.stack([zpost(p) for p in setlist])                                            # [n, d]
    # do-op baseline: unsupervised pred Δ on shared genes
    @torch.no_grad()
    def doop_delta(p):
        gi = [g2i[t] for t in p]
        return (op.decode(op.apply_sets(eps0, [gi], A))[0].cpu().numpy() - ctrl_recon)[sidx]
    D = np.stack([doop_delta(p) for p in setlist])                                       # [n, |shared|]

    def per_pair_corr(P, idx):
        return np.array([cc(P[i], Y[i]) for i in idx])

    def eval_split(train_idx, test_idx, tag):
        rg = Ridge(alpha=a.alpha).fit(Z[train_idx], Y[train_idx])
        pred = rg.predict(Z[test_idx])
        read_c = np.array([cc(pred[k], Y[test_idx[k]]) for k in range(len(test_idx))])
        doop_c = per_pair_corr(D, test_idx)
        print(f"  [{tag}] n_test={len(test_idx)} | READOUT held-out per-pair corr median={np.nanmedian(read_c):+.3f} "
              f"mean={np.nanmean(read_c):+.3f} | do-op baseline median={np.nanmedian(doop_c):+.3f}", flush=True)
        return read_c

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(setlist)); nt = int(0.8 * len(perm))
    eval_split(perm[:nt], perm[nt:], 'random 80/20')
    # leave-TF-out: hold out ALL pairs containing a held-out TF (strict generalization)
    for hotf in ['ETV2', 'TAL1', 'GATA2']:
        test_idx = np.array([i for i, p in enumerate(setlist) if hotf in p])
        train_idx = np.array([i for i in range(len(setlist)) if hotf not in setlist[i]])
        if len(test_idx) >= 5:
            eval_split(train_idx, test_idx, f'leave-{hotf}-out')

    # ---------- (c) DIRECTION readout — the CORRECT cross-dataset metric (day7 vs day14/28 => raw Δ not
    # comparable; fate-axis DIRECTION is). z_post(combo) -> observed Briggs combo fate projection rho_briggs. ----
    print("\n===== (c) DIRECTION readout: z_post(combo) -> Briggs fate projection rho_briggs (leave-TF-out) =====", flush=True)
    import pandas as pd
    from scipy.stats import spearmanr
    oc = pd.read_csv(f'{EXP20}/results/briggs/observed_combo_targets.csv')
    rows = []
    for _, r in oc.iterrows():
        tfs = [t for t in str(r['combo']).split('|') if t in g2i]
        if len(tfs) >= 1 and len(tfs) == len(str(r['combo']).split('|')):
            rows.append((tfs, float(r['rho_briggs']), int(r['k'])))
    Zc = np.stack([zpost(t) for t, _, _ in rows]); yc = np.array([y for _, y, _ in rows]); kc = np.array([k for _, _, k in rows])
    print(f"  n={len(rows)} observed Briggs sets (k1..) with all TFs in WM vocab", flush=True)
    rng2 = np.random.default_rng(0); pm = rng2.permutation(len(rows)); ntr = int(0.8 * len(pm))
    rg = Ridge(alpha=a.alpha).fit(Zc[pm[:ntr]], yc[pm[:ntr]])
    pr = rg.predict(Zc[pm[ntr:]])
    print(f"  [random 80/20] Spearman(pred rho, obs rho) = {spearmanr(pr, yc[pm[ntr:]]).correlation:+.3f}", flush=True)
    for hotf in ['ETV2', 'TAL1', 'GATA2', 'LMO2']:
        te = np.array([i for i, (t, _, _) in enumerate(rows) if hotf in t])
        tr = np.array([i for i in range(len(rows)) if hotf not in rows[i][0]])
        if len(te) >= 5:
            rg = Ridge(alpha=a.alpha).fit(Zc[tr], yc[tr]); pr = rg.predict(Zc[te])
            print(f"  [leave-{hotf}-out] n_test={len(te)} Spearman={spearmanr(pr, yc[te]).correlation:+.3f}", flush=True)
    print("  (compare: raw fate-axis projection baseline k=1 ≈ +0.32; supervised fate readout ≈ +0.71)", flush=True)

    if a.save_readout:                                                   # save readout trained on the 80% TRAIN split
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Zc[pm[:ntr]])
        rr = Ridge(alpha=a.alpha).fit(sc.transform(Zc[pm[:ntr]]), yc[pm[:ntr]])
        ho_sp = spearmanr(rr.predict(sc.transform(Zc[pm[ntr:]])), yc[pm[ntr:]]).correlation
        np.savez(a.save_readout, coef=rr.coef_, intercept=rr.intercept_, mean=sc.mean_, scale=sc.scale_, target='rho_briggs')
        print(f"  saved held-out-combo-trained readout -> {a.save_readout} (held-out Spearman {ho_sp:+.3f})", flush=True)
    if a.save_feats:                                                     # observed-combo latent manifold for Mahalanobis support
        combo_names = np.array(['|'.join(t) for t, _, _ in rows])
        np.savez(a.save_feats, Z=Zc.astype(np.float32), combo=combo_names)
        print(f"  saved support manifold ({Zc.shape}) -> {a.save_feats}", flush=True)
    print("\n[readout-combo done]", flush=True)


if __name__ == '__main__':
    main()
