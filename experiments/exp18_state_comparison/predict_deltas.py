"""Stage 1 of the exp13/exp16-vs-STATE comparison: predict each held-out K562 single
perturbation's gene-space DELTA on STATE's 2000-HVG axis, using the causal world model's
shift encoder + do-operator (zero-shot for the pert label).

Reuses the experiment's own model/loader code (passed via --exp_dir) so the architecture
matches its checkpoint. Output: an .npz with {perts, delta[n_perts,2000], state_hvg} that
Stage 2 (score_celleval.py, state_env) anchors to STATE's real controls and scores with
the identical cell_eval.MetricsEvaluator STATE was scored with.

  exp13 (exp8 ckpt): additive shift   z_target = z_prev + (I−Aᵀ)⁻¹S ; no modality
  exp16:             mutilation u-score + cell-line modality (k562)

Run (base env), e.g. exp16:
  python predict_deltas.py --exp_dir ../exp16_multicellline \
    --checkpoint ../exp16_multicellline/results/grid/exp16_multicellline/checkpoints/phase1_best.pt \
    --state_hvg results/state_hvg.json --out results/exp16_delta.npz
"""
import os, sys, json, argparse
import numpy as np
import torch
import scipy.sparse


def _solve(LHS, rhs):
    return torch.linalg.solve(LHS, rhs.T).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_dir', required=True, help='experiment dir (its evaluation/data on path)')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--state_hvg', required=True, help='json list of STATE 2000 HVG gene names')
    ap.add_argument('--perts', default=None, help='json list of perts; default = the test list file')
    ap.add_argument('--cell_file', default='/large_storage/ctc/userspace/aadduri/revisions/replogle_unfiltered/K562_essential_raw_singlecell_01.h5ad')
    ap.add_argument('--n_ctrl_cells', type=int, default=500)
    ap.add_argument('--max_cells', type=int, default=200000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.exp_dir))
    from evaluation.run_shift_eval_norman import load_shift_model
    from data.load_replogle_celline_aligned import load_replogle_celline_aligned
    dev = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, gene_names, ck = load_shift_model(args.checkpoint, dev)
    gidx = {g: i for i, g in enumerate(gene_names)}
    d = model.latent_dim
    cfg = ck.get('config', {}) or {}
    p1 = cfg.get('phase1', {})
    is_mut = 'pert_mutilate_u_mode' in p1
    U_MODE = p1.get('pert_mutilate_u_mode', 'shift'); U_MAX = float(p1.get('pert_mutilate_u_max', 0.5))
    BETA = float(p1.get('pert_mutilate_beta', 1.0))
    celline_files = cfg['data'].get('celline_files')
    has_mod = celline_files is not None
    mod = (list(celline_files).index('k562') if has_mod else None)
    print(f"[predict] genes={len(gene_names)} latent={d} mutilation={is_mut} modality={has_mod} (k562 mod={mod})", flush=True)

    state_hvg = json.load(open(args.state_hvg))
    common = [g for g in state_hvg if g in gidx]
    print(f"[predict] STATE HVG in model panel: {len(common)}/{len(state_hvg)}", flush=True)

    dd = load_replogle_celline_aligned(args.cell_file, target_gene_names=gene_names, cell_line='k562',
                                       max_cells=args.max_cells, min_cells_per_pert=5)
    X = dd['adata'].X; X = (X.toarray() if scipy.sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    ctrl_idx = np.array(dd['control_idx'])
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ctrl_idx), min(args.n_ctrl_cells, len(ctrl_idx)), replace=False)
    x_ctrl = torch.tensor(X[ctrl_idx[sel]], dtype=torch.float32, device=dev)

    def emb(x):
        if has_mod:
            mm = torch.full((x.shape[0],), mod, dtype=torch.long, device=dev)
            return model.encoder(x, modality_id=mm)
        return model.encoder(x)

    with torch.no_grad():
        out = emb(x_ctrl); eps_ctrl = out[2]
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        LHS = torch.eye(d, device=dev, dtype=A.dtype) - A.T
        z_prev = torch.clamp(_solve(LHS, eps_ctrl), -20, 20)
        ctrl_recon = model.decoder(z_prev).mean(0).cpu().numpy()

    if args.perts:
        perts = json.load(open(args.perts))
    else:
        perts = sorted(dd['single_perts'].keys())
    perts = [p for p in perts if p in gidx]            # pert gene must be on model axis
    print(f"[predict] predicting {len(perts)} perts", flush=True)

    deltas = np.zeros((len(perts), len(common)), np.float32)
    common_mi = np.array([gidx[g] for g in common])
    for k, g in enumerate(perts):
        gi = gidx[g]
        with torch.no_grad():
            if has_mod:
                S = model.compute_shift(torch.tensor([gi], device=dev),
                                        modality_id=torch.tensor([mod], device=dev)).sum(0)
            else:
                S = model.compute_shift(torch.tensor([gi], device=dev)).sum(0)
            z_S = torch.clamp(_solve(LHS, S.unsqueeze(0)), -20, 20)
            z_target = z_prev + z_S
            if is_mut:
                u = (BETA * S.abs()).clamp(max=1.0) if U_MODE == 'shift' else \
                    model.get_loadings()[torch.tensor([gi], device=dev)].sum(0).clamp(0, U_MAX)
                eps_tilde = u.unsqueeze(0) * z_target + (1 - u).unsqueeze(0) * eps_ctrl
                z_post, _, _ = model.dag.scm_intervene_u(eps_ctrl, A, eps_tilde, u)
                pred = model.decoder(z_post).mean(0).cpu().numpy() - ctrl_recon
            else:
                pred = model.decoder(z_target).mean(0).cpu().numpy() - ctrl_recon  # additive shift_add
        deltas[k] = pred[common_mi]
        if k % 100 == 0: print(f"  {k}/{len(perts)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, perts=np.array(perts), delta=deltas, state_hvg=np.array(common))
    print(f"saved → {args.out}  ({deltas.shape})")


if __name__ == '__main__':
    main()
