"""Track 2 — Briggs-adapted READOUT HEAD on the FROZEN causal WM.

Question: the raw fate-axis projection of the one-step do-op loses fidelity at high k (~0 by k>=7).
Does the WM's LATENT counterfactual z_post still CARRY the endpoint-fate signal, recoverable by a small
supervised head trained on a HELD-OUT Briggs combo split? If yes -> we can score cocktails for RL without
corrupting the causal WM (only the readout is learned; causal core stays frozen).

Feature per combo = frozen WM latent z_post (mean over control cells) from the do_mean do-op (interaction
head included). Target = observed endpoint fate rho_obs (Cao axis; Briggs axis as secondary). Split by
combo (random 80/20, seed 0) + optional strict leave-TF-out. Compare readout (Ridge / MLP) vs the RAW
fate-axis projection baseline, OVERALL and BY k (the high-k buckets are the real test).

Caches features to results/briggs/readout_feats_<tag>.npz (extraction is the slow part; training iterates fast).
Run: python evaluation/briggs/readout_head.py --ckpt results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt
"""
import os, sys, csv, argparse, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from evaluation.run_shift_eval_norman import load_shift_model
from evaluation.briggs.hspc_sim import build_cao_hsc_signature
from data.load_joung import load_joung
from scipy.stats import spearmanr
torch.set_grad_enabled(False)
V = "/large_storage/goodarzilab/shervin/CausalBioRL/Joung_TF_Atlas/vocab_v2"


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def extract(ckpt, answer, n_ctrl, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"[feats] loaded cache {cache}: Z{d['Z'].shape}", flush=True)
        return d
    model, gene_names, _ = load_shift_model(ckpt, 'cpu')
    gn = list(map(str, gene_names)); g2i = {g: i for i, g in enumerate(gn)}
    dd = load_joung(proc_path=V + "/joung_singles_proc_v2b.h5ad", n_hvg=None,
                    effective_tfs=V + "/effective_tf_genes.txt", split_seed=42, train_frac=0.9, val_frac=0.0)
    Xc = dd['adata'].X; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    ctrl = np.array(dd['control_idx']); rng = np.random.default_rng(0)
    xc = torch.tensor(Xc[ctrl[rng.choice(len(ctrl), min(n_ctrl, len(ctrl)), replace=False)]])
    _, _, eps = model.encoder(xc); A = model.dag.sample_A(1).squeeze(0)
    z_prev = model.dag.scm_forward(eps, A); cr = model.decoder(z_prev).mean(0)
    d_cao, cidx = build_cao_hsc_signature(gn)
    reliable = set(open(V + "/effective_tf_genes.txt").read().split())
    obs = list(csv.DictReader(open(answer)))
    Z, deltas, meta = [], [], []
    for r in obs:
        tfs = r['combo'].split('|'); gi = [g2i[g] for g in tfs if g in g2i]
        if len(gi) != len(tfs): continue
        z_post, _ = model.intervene_shift_mutilate(eps, gi, A, z_prev, u_mode='shift', shift_beta=1.0)
        Z.append(z_post.mean(0).cpu().numpy())                                   # frozen WM latent feature [d]
        deltas.append((model.decoder(z_post).mean(0) - cr).cpu().numpy())        # for the raw baseline
        meta.append((r['combo'], int(r['k']), float(r['rho_cao']), float(r['rho_briggs']),
                     int(r['is_P1']), sum(t in reliable for t in tfs)))
    Z = np.stack(Z).astype(np.float32); D = np.stack(deltas); D = D - D.mean(0)  # drift-center
    rho_raw = np.array([cos(D[i][cidx], d_cao) for i in range(len(D))], np.float32)
    combo = np.array([m[0] for m in meta]); k = np.array([m[1] for m in meta])
    rc = np.array([m[2] for m in meta], np.float32); rb = np.array([m[3] for m in meta], np.float32)
    isp1 = np.array([m[4] for m in meta]); nrel = np.array([m[5] for m in meta])
    np.savez_compressed(cache, Z=Z, rho_raw=rho_raw, combo=combo, k=k, rho_cao=rc, rho_briggs=rb, is_P1=isp1, n_reliable=nrel)
    print(f"[feats] extracted {len(Z)} combos -> {cache}", flush=True)
    return np.load(cache, allow_pickle=True)


def byk_report(name, pred, y, k, base):
    print(f"  {name}: test Spearman overall {spearmanr(pred, y).correlation:+.3f}  (raw baseline {spearmanr(base, y).correlation:+.3f})", flush=True)
    for kk in [2, 3, 4, 5, 6, 7]:
        m = k == kk
        if m.sum() > 15:
            print(f"     k={kk} (n={int(m.sum()):4d}): readout {spearmanr(pred[m], y[m]).correlation:+.3f}  |  raw {spearmanr(base[m], y[m]).correlation:+.3f}", flush=True)
    hi = k >= 7
    if hi.sum() > 15:
        print(f"     k>=7 (n={int(hi.sum()):4d}): readout {spearmanr(pred[hi], y[hi]).correlation:+.3f}  |  raw {spearmanr(base[hi], y[hi]).correlation:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/grid/v2b_deg3_t25_frozen_r32_s1/checkpoints/phase1_epoch40.pt')
    ap.add_argument('--answer', default='results/briggs/observed_combo_targets.csv')
    ap.add_argument('--n_ctrl', type=int, default=128); ap.add_argument('--target', default='rho_cao')
    ap.add_argument('--holdout', default='random', choices=['random', 'loto', 'holdhits'])
    ap.add_argument('--exclude_tfs', default='', help="comma TFs held out of the FIT (leave-hit-out; test = combos containing them)")
    a = ap.parse_args()
    tag = a.ckpt.split('results/grid/')[1].split('/')[0]
    ansb = os.path.splitext(os.path.basename(a.answer))[0].replace('observed_combo_targets', 'ans')
    d = extract(a.ckpt, a.answer, a.n_ctrl, f'results/briggs/readout_feats_{tag}_{ansb}.npz')
    Z, k, y, raw = d['Z'], d['k'], d[a.target], d['rho_raw']
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    if a.holdout == 'random':
        idx = rng.permutation(len(Z)); cut = int(0.8 * len(Z)); tr, te = idx[:cut], idx[cut:]
        print(f"\n===== READOUT ({tag}) target={a.target} | random 80/20 split =====", flush=True)
    elif a.holdout == 'holdhits':  # leave-HIT-out: hold out combos containing any specified TF (no leakage)
        combos = d['combo']; ho = set(t for t in a.exclude_tfs.split(',') if t)
        te = np.array([i for i, c in enumerate(combos) if any(t in ho for t in c.split('|'))])
        tr = np.array([i for i in range(len(Z)) if i not in set(te.tolist())])
        print(f"\n===== READOUT ({tag}) target={a.target} | leave-HIT-out ({len(ho)} TFs held from fit, "
              f"{len(te)} hit-containing test combos) =====", flush=True)
    else:  # strict: hold out all combos containing any of 20 random held-out TFs (true generalization)
        combos = d['combo']; allt = sorted({t for c in combos for t in c.split('|')})
        ho = set(rng.choice(allt, max(1, len(allt) // 5), replace=False))
        te = np.array([i for i, c in enumerate(combos) if any(t in ho for t in c.split('|'))])
        tr = np.array([i for i in range(len(Z)) if i not in set(te.tolist())])
        print(f"\n===== READOUT ({tag}) target={a.target} | leave-TF-out ({len(ho)} TFs, {len(te)} test combos) =====", flush=True)

    sc = StandardScaler().fit(Z[tr]); Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
    ridge = Ridge(alpha=10.0).fit(Ztr, y[tr]); pr = ridge.predict(Zte)
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True, random_state=0).fit(Ztr, y[tr]); pm = mlp.predict(Zte)
    print(f"  baseline: raw fate-axis projection test Spearman = {spearmanr(raw[te], y[te]).correlation:+.3f}", flush=True)
    byk_report('Ridge(z_post->fate)', pr, y[te], k[te], raw[te])
    byk_report('MLP(z_post->fate)  ', pm, y[te], k[te], raw[te])
    # save Ridge readout for RL reuse
    if a.holdout == 'random':                                                  # deployed: refit on ALL combos
        scf = StandardScaler().fit(Z); rf = Ridge(alpha=10.0).fit(scf.transform(Z), y)
        out = f'results/briggs/readout_ridge_{tag}_{ansb}_{a.target}.npz'      # ansb keeps backgrounds separate
        np.savez(out, coef=rf.coef_, intercept=rf.intercept_, mean=scf.mean_, scale=scf.scale_, target=a.target)
        print(f"wrote {out}", flush=True)
    elif a.holdout == 'holdhits':                                              # leak-free: fit ONLY on clean (non-hit) combos
        out = f'results/briggs/readout_ridge_{tag}_{a.target}_holdhits.npz'
        np.savez(out, coef=ridge.coef_, intercept=ridge.intercept_, mean=sc.mean_, scale=sc.scale_, target=a.target)
        print(f"wrote {out}  (fit on {len(tr)} non-hit combos)", flush=True)


if __name__ == '__main__':
    main()
