"""Gradient-flow smoke — where does the perturbation-prediction learning signal actually go?

Loads a trained checkpoint, runs the structural do-operator (TRAIN mode → soft Concrete gate,
so grads reach dag.theta) on a batch of TFs, computes the drift-removed pert-Δ MSE to the
observed LFC, backprops ONCE, and reports the gradient norm on each component (and normalized by
that component's param norm = relative sensitivity of the pert loss to each block).

If the pert loss is nearly flat in A (tiny dag grad) but steep in D / shift_encoder, that is the
mechanistic reason A is not load-bearing while shift+decoder are: SGD has no pressure to use A.

Usage: python evaluation/gradient_flow.py --checkpoint <ckpt> [--n_tf 60 --n_ctrl 400]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp, anndata as ad
from evaluation.run_shift_eval_norman import load_shift_model
from data.load_joung import load_joung

PROC = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'
GROUPS = ['encoder', 'shift_encoder.gate_out', 'shift_encoder', 'dag', 'decoder.D', 'decoder.b', 'shift_scale']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--n_tf', type=int, default=60)
    ap.add_argument('--n_ctrl', type=int, default=400)
    a = ap.parse_args()
    model, gene_names, ck = load_shift_model(a.checkpoint, 'cpu')
    p1 = ck.get('config', {}).get('phase1', {}) or {}
    u_mode = p1.get('pert_mutilate_u_mode', 'shift'); beta = float(p1.get('pert_mutilate_beta', 1.0))
    g2i = {g.upper(): i for i, g in enumerate(gene_names)}
    model.train()                                     # soft Concrete gate → differentiable A(theta)
    torch.set_grad_enabled(True)

    dd = load_joung(n_hvg=None); de = dd['de_single']
    test = [g for g in dd['pert_split']['test'] if g in de and g.upper() in g2i][:a.n_tf]
    train = [g for g in dd['pert_split']['train'] if g in de]
    drift = torch.tensor(np.mean([de[g] for g in train], 0), dtype=torch.float32)
    obs = torch.tensor(np.stack([np.asarray(de[g], np.float32) for g in test]))

    A_read = ad.read_h5ad(PROC, backed='r')
    is_ctrl = (A_read.obs['is_control'].astype(str) == 'True').values
    rng = np.random.default_rng(0)
    cr = np.sort(rng.choice(np.where(is_ctrl)[0], min(a.n_ctrl, is_ctrl.sum()), replace=False))
    Xc = A_read.X[cr]; Xc = torch.tensor((Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32))

    eps_ctrl = model.encode(Xc)[2]
    preds = []
    for g in test:
        gi = g2i[g.upper()]
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        z_prev = model.dag.scm_forward(eps_ctrl, A)
        z_post, _ = model.intervene_shift_mutilate(eps_ctrl, [gi], A, z_prev, u_mode=u_mode, shift_beta=beta)
        preds.append((model.decoder(z_post) - model.decoder(z_prev)).mean(0))
    pred = torch.stack(preds)
    loss = ((pred - drift) - (obs - drift)).pow(2).mean()
    model.zero_grad()
    loss.backward()

    def gnorm(prefix):
        tot = sum(p.grad.detach().pow(2).sum().item() for n, p in model.named_parameters()
                  if p.grad is not None and (n == prefix or n.startswith(prefix + '.')))
        return tot ** 0.5

    def pnorm(prefix):
        tot = sum(p.detach().pow(2).sum().item() for n, p in model.named_parameters()
                  if n == prefix or n.startswith(prefix + '.'))
        return tot ** 0.5

    print(f"=== GRADIENT FLOW | {os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))} "
          f"| pert-Δ MSE over {len(test)} held-out TFs | loss={loss.item():.4f} ===")
    print(f"  {'component':24s} {'grad_norm':>12s} {'param_norm':>12s} {'grad/param':>12s}")
    rows = []
    for gp in GROUPS:
        gnp = gnorm(gp); pnp = pnorm(gp); rows.append((gp, gnp, pnp, gnp / (pnp + 1e-12)))
    # shift_encoder above double-counts gate_out; report trunk-only too
    for gp, gn, pn, r in rows:
        print(f"  {gp:24s} {gn:12.5f} {pn:12.4f} {r:12.6f}")
    tot = sum(r[1] for r in rows if r[0] not in ('shift_encoder.gate_out',)) or 1e-9
    print("  --- share of total grad-norm (relative learning pressure) ---")
    for gp, gn, pn, r in rows:
        if gp == 'shift_encoder.gate_out': continue
        print(f"  {gp:24s} {100*gn/tot:6.1f}%")


if __name__ == '__main__':
    main()
