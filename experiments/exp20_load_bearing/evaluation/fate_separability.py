"""FAITHFUL fate-separability check — uses the SAME cosine-to-mouse-signature method that
MADE the labels (fate_space: Cao-2019 mouse organogenesis deviation signatures, shared-PC
removed, cosine β), not a generic kNN. Answers, on HELD-OUT single TFs:

  - intrinsic difficulty: the obs-β argmax MARGIN (top1−top2) + specificity → are fates
    low-margin (so top-1 is brittle) or cleanly separated?
  - data ceiling vs model: obs-β (label) vs WM-pred-β recovery — top1 / top3 / top5 /
    β-cosine, and fate-space retrieval (is pred-β_i nearest obs-β_i among held-out TFs).
top-k & β-cosine matter because top-1 over a low-margin 20-way call is the hardest framing.

Run (cpu_high_mem):  python evaluation/fate_separability.py --checkpoint <ckpt> --device cpu
"""
import os, sys, json, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from evaluation.heldout_zeroshot_eval import fate_space
from data.load_joung import _lognorm, DEFAULT_PROC, split_single_perts
from evaluation.run_shift_eval_norman import load_shift_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--min_cells', type=int, default=50)
    ap.add_argument('--tag', default='ugate')
    ap.add_argument('--output', default='results/fate_separability')
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    dev = torch.device(a.device); torch.set_grad_enabled(False)

    model, gn, _ = load_shift_model(a.checkpoint, dev); model.eval()
    g2i = {g: i for i, g in enumerate(gn)}

    import anndata as ad, scipy.sparse as sp
    S = ad.read_h5ad(DEFAULT_PROC); Xs = _lognorm(S.X)
    isc = S.obs['is_control'].values.astype(bool); ctrl_idx = np.where(isc)[0]
    tfg = S.obs['TF_gene'].astype(str).values
    ctrl_mean = np.asarray(Xs[ctrl_idx].mean(0)).ravel().astype(np.float32)
    perts = {}
    for i in np.where(~isc)[0]:
        g = tfg[i]
        if g in g2i:
            perts.setdefault(g, []).append(i)
    perts = {g: v for g, v in perts.items() if len(v) >= a.min_cells}
    split = split_single_perts(perts, train_frac=0.8, val_frac=0.1, seed=42)
    train_g = [g for g in split['train'] if g in perts]
    test_g = [g for g in split['test'] if g in perts]

    # observed Δ (sparse slices) for train (nuisance fit) + held-out test
    train_obs = np.stack([np.asarray(Xs[np.asarray(perts[g])].mean(0)).ravel() - ctrl_mean for g in train_g])
    obs = np.stack([np.asarray(Xs[np.asarray(perts[g])].mean(0)).ravel() - ctrl_mean for g in test_g])

    # WM-predicted single-TF Δ for held-out TFs
    model.dag.eval(); A = model.dag.sample_A(n_samples=1).squeeze(0)
    selc = ctrl_idx[np.random.default_rng(0).choice(len(ctrl_idx), min(300, len(ctrl_idx)), replace=False)]
    xc = torch.tensor(np.asarray(Xs[selc].todense(), np.float32), device=dev)
    _, _, eps_ctrl = model.encoder(xc); eps0 = eps_ctrl.mean(0, keepdim=True)
    I = torch.eye(model.latent_dim, device=dev)
    z_prev = torch.linalg.solve(I - A.T, eps0.T).T
    ctrl_recon = model.decoder(z_prev).squeeze(0).cpu().numpy()
    pred = []
    for g in test_g:
        sh = model.compute_shift(torch.tensor([g2i[g]], device=dev)).sum(0)
        zS = torch.linalg.solve(I - A.T, sh.unsqueeze(1)).T
        pred.append(model.decoder(z_prev + zS).squeeze(0).cpu().numpy() - ctrl_recon)
    pred = np.array(pred, np.float32)

    # FAITHFUL fate deconvolution (cosine vs mouse signatures, nuisance fit on train)
    res, bp, bo, top1, bcos, *_ = fate_space(pred, obs, train_obs, gn)
    C = bo.shape[1]

    def topk(k):
        return float(np.mean([bo[i].argmax() in set(bp[i].argsort()[::-1][:k]) for i in range(len(bp))]))

    # intrinsic difficulty: margin of the OBSERVED-β argmax (top1 − top2)
    so = np.sort(bo, 1)[:, ::-1]
    margin = so[:, 0] - so[:, 1]
    # fate-space retrieval: is pred-β_i nearest obs-β_i among held-out TFs?
    Pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    On = bo / (np.linalg.norm(bo, axis=1, keepdims=True) + 1e-8)
    Cmat = Pn @ On.T
    retr = float(np.mean([Cmat[i].argmax() == i for i in range(len(bp))]))

    out = {'tag': a.tag, 'n_test': len(test_g), 'n_celltypes': C, 'chance_top1': round(1.0 / C, 3),
           'fate_top1': round(res['fate_top1'], 3), 'fate_top3': round(topk(3), 3),
           'fate_top5': round(topk(5), 3), 'fate_betacos': round(res['fate_betacos'], 3),
           'fate_retrieval_top1': round(retr, 3), 'chance_retrieval': round(1.0 / len(test_g), 3),
           'obs_margin_median': round(float(np.median(margin)), 3),
           'obs_margin_mean': round(float(margin.mean()), 3),
           'frac_obs_margin_gt_0.1': round(float((margin > 0.1).mean()), 3)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(f'{a.output}/{a.tag}_summary.json', 'w'), indent=2)
    print(f"saved → {a.output}/{a.tag}_summary.json")


if __name__ == '__main__':
    main()
