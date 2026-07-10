"""Localize the discrimination failure of the Joung shift encoder. Three diagnostics:

D1. TRAIN vs HELD-OUT retrieval — retrieval_top1 + discrimination_cos on TRAIN TFs vs TEST.
    Fork: train high / test chance ⇒ OOD-extrapolation collapse (generalization failure);
    train also low ⇒ the SCM+linear-decoder can't represent TF-specific effects at all.
D2. SHIFT VARIANCE, train vs test — spread of the latent shift S^g = ShiftEncoder(label).
    Held-out shifts bunched near the mean (low variance, high pairwise cosine) ⇒ MLP maps
    OOD fingerprints to a generic output (classic extrapolation pathology).
D3. IS THE FINGERPRINT INFORMATIVE? — fit Ridge from the R^50 control-coexpression
    fingerprint directly to observed Δ on TRAIN, retrieval on TEST. At chance ⇒ the input
    representation itself carries no TF-identity (no encoder can fix it).

Retrieval_top1 chance = 1/N (we subsample train to the test N for a fair top1); discrimination
_cos chance ≈ 0.5 regardless of N. Run (high-mem node):
  python evaluation/shift_diagnostics.py --checkpoint .../phase1_epoch80.pt --tag epoch80
"""
import os, sys, json, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.heldout_zeroshot_eval import zscore_rows, _solve


def retrieval(pred, obs):
    """pred,obs [T,G] aligned. Returns (retrieval_top1, discrimination_cos)."""
    Pz, Oz = zscore_rows(pred), zscore_rows(obs)
    C = Pz @ Oz.T; T = C.shape[0]
    top1 = float((C.argmax(1) == np.arange(T)).mean())
    diag = np.diag(C)[:, None]
    disc = float((((diag > C).sum(1)) / (T - 1)).mean())
    return top1, disc


def predict_shifts_and_delta(model, gene_names, tfs, gidx, z_prev, ctrl_recon, eps_ctrl, A, LHS, dev,
                             u_mode, beta, u_max):
    """Return latent shifts S [T,d] and predicted Δ [T,G] for a list of TFs (do-operator)."""
    d = model.latent_dim; G = len(gene_names)
    S_mat = np.zeros((len(tfs), d), np.float32); D_mat = np.zeros((len(tfs), G), np.float32)
    for k, g in enumerate(tfs):
        gi = gidx[g]
        S = model.compute_shift(torch.tensor([gi], device=dev)).sum(0)        # [d] latent shift
        S_mat[k] = S.detach().cpu().numpy()
        z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
        z_target = z_prev + z_S
        u = (beta * S.abs()).clamp(max=1.0) if u_mode == 'shift' else \
            model.get_loadings()[torch.tensor([gi], device=dev)].sum(0).clamp(0, u_max)
        eps_tilde = u.unsqueeze(0) * z_target + (1 - u).unsqueeze(0) * eps_ctrl
        z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
        D_mat[k] = (model.decoder(z_post).mean(0) - ctrl_recon).detach().cpu().numpy()
    return S_mat, D_mat


def shift_stats(S):
    """spread statistics of a set of latent shift vectors S [T,d]."""
    mu = S.mean(0);
    Sz = zscore_rows(S)
    pair = (Sz @ Sz.T); off = (pair.sum() - np.trace(pair)) / (len(S) * (len(S) - 1))
    return {'total_var': float(S.var(0).sum()), 'mean_norm': float(np.linalg.norm(S, axis=1).mean()),
            'norm_std': float(np.linalg.norm(S, axis=1).std()),
            'mean_dist_to_centroid': float(np.linalg.norm(S - mu, axis=1).mean()),
            'mean_pairwise_cos': float(off)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n_train', type=int, default=400, help='train TFs to subsample')
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'shift_diagnostics', args.tag); os.makedirs(out, exist_ok=True)
    dev = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_grad_enabled(False)

    from evaluation.run_shift_eval_norman import load_shift_model
    from data.load_joung import load_joung
    import scipy.sparse as sp
    model, gene_names, ck = load_shift_model(args.checkpoint, dev)
    gidx = {g: i for i, g in enumerate(gene_names)}; d = model.latent_dim
    p1 = ck.get('config', {}).get('phase1', {})
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift'); BETA = float(p1.get('pert_mutilate_beta', 1.0))
    U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))

    dd = load_joung(n_hvg=None); de = dd['de_single']
    Xc = dd['adata'].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    test = [g for g in dd['pert_split']['test'] if g in de and g in gidx]
    train_all = [g for g in dd['pert_split']['train'] if g in de and g in gidx]
    rng = np.random.default_rng(0)
    train = list(rng.choice(train_all, min(args.n_train, len(train_all)), replace=False))
    N = len(test)
    train_match = list(rng.choice(train, N, replace=False))   # matched-N subset for fair top1
    print(f"[diag] test={len(test)} train_sample={len(train)} (matched-N={N})", flush=True)

    ctrl = np.array(dd['control_idx']); sel = ctrl[rng.choice(len(ctrl), min(args.n_ctrl_cells, len(ctrl)), replace=False)]
    x_ctrl = torch.tensor(Xc[sel], dtype=torch.float32, device=dev)
    _, _, eps_ctrl = model.encoder(x_ctrl)
    A = model.dag.sample_A(n_samples=1).squeeze(0)
    LHS = torch.eye(d, device=dev, dtype=A.dtype) - A.T
    z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
    ctrl_recon = model.decoder(z_prev).mean(0)

    def pred(tfs):
        return predict_shifts_and_delta(model, gene_names, tfs, gidx, z_prev, ctrl_recon, eps_ctrl, A, LHS, dev,
                                        U_MODE, BETA, U_MAX)
    S_tr, D_tr = pred(train); S_te, D_te = pred(test)
    O_tr = np.stack([np.asarray(de[g], np.float32) for g in train])
    O_te = np.stack([np.asarray(de[g], np.float32) for g in test])
    # matched-N train arrays
    mi = [train.index(g) for g in train_match]
    D_trm, O_trm = D_tr[mi], O_tr[mi]

    # D1 — retrieval train vs test
    tr_top1, tr_disc = retrieval(D_trm, O_trm)
    te_top1, te_disc = retrieval(D_te, O_te)

    # D2 — shift variance train vs test
    st_tr, st_te = shift_stats(S_tr), shift_stats(S_te)

    # D3 — ridge from fingerprint (model pert_label, the control-coexp R^50) → observed Δ
    fp = model.pert_label.detach().cpu().numpy()              # [G, label_dim] (≥50)
    Xtr = fp[[gidx[g] for g in train]]; Xte = fp[[gidx[g] for g in test]]
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=10.0).fit(Xtr, O_tr)
    Dhat_te = ridge.predict(Xte); Dhat_trm = ridge.predict(fp[[gidx[g] for g in train_match]])
    rg_te_top1, rg_te_disc = retrieval(Dhat_te, O_te)
    rg_tr_top1, rg_tr_disc = retrieval(Dhat_trm, O_trm)
    fp_dim = fp.shape[1]

    summary = {
        'tag': args.tag, 'epoch': ck.get('epoch'), 'n_test': N, 'n_train_sample': len(train),
        'chance_top1_matchedN': 1.0 / N,
        'D1_model': {'train_top1': tr_top1, 'train_disc': tr_disc, 'test_top1': te_top1, 'test_disc': te_disc},
        'D2_shift_var': {'train': st_tr, 'test': st_te,
                         'var_ratio_test_over_train': st_te['total_var'] / (st_tr['total_var'] + 1e-9)},
        'D3_ridge_fingerprint': {'fingerprint_dim': fp_dim,
                                 'train_top1': rg_tr_top1, 'train_disc': rg_tr_disc,
                                 'test_top1': rg_te_top1, 'test_disc': rg_te_disc},
    }
    json.dump(summary, open(f"{out}/summary.json", 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nsaved → {out}/summary.json")
    print("\n=== INTERPRETATION ===")
    print(f"D1: model train_top1={tr_top1:.3f} disc={tr_disc:.3f}  |  test_top1={te_top1:.3f} disc={te_disc:.3f} "
          f"(chance top1={1.0/N:.3f}, disc=0.5)")
    print(f"D2: shift var test/train = {summary['D2_shift_var']['var_ratio_test_over_train']:.3f}; "
          f"pairwise-cos train={st_tr['mean_pairwise_cos']:.3f} test={st_te['mean_pairwise_cos']:.3f}")
    print(f"D3: ridge fingerprint→Δ  test_top1={rg_te_top1:.3f} disc={rg_te_disc:.3f} (train_top1={rg_tr_top1:.3f})")


if __name__ == '__main__':
    main()
