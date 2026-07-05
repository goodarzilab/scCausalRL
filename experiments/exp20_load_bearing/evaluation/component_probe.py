"""Component load-bearing probe — extends agap_probe to A vs DECODER vs SHIFT, all in the
SAME 1−cos(Δ_real, Δ_variant) metric so they are directly comparable to probe_Agap.

For N held-out TFs, do-operator Δ with the real model, then re-run perturbing ONE component:
  probe_A       : A ≡ 0            (model.dag.disabled) — how much A controls the output
  probe_Dperm   : decoder gene-rows permuted
  probe_Drand   : decoder := random orthonormal, matched ‖·‖_F
  probe_shiftID : this TF's do-op run with ANOTHER TF's shift — how much perturbation IDENTITY matters
  crossTF_div   : mean 1−cos between distinct TFs' real Δ — mode-collapse check (0 ⇒ all preds alike)
Higher 1−cos ⇒ that component controls more of the prediction. Decoder-absorption hypothesis
predicts probe_Dperm ≫ probe_A (the gene-space structure lives in D; A is a minor add-on).

Usage: python evaluation/component_probe.py --checkpoint <ckpt> [--n_tf 24 --n_ctrl 500]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp, anndata as ad
from evaluation.run_shift_eval_norman import load_shift_model

PROC = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def cos(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--joung', default=PROC)
    ap.add_argument('--n_tf', type=int, default=24)
    ap.add_argument('--n_ctrl', type=int, default=500)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, gene_names, ck = load_shift_model(a.checkpoint, 'cpu')
    p1 = ck.get('config', {}).get('phase1', {}) or {}
    u_mode = p1.get('pert_mutilate_u_mode', 'shift'); beta = float(p1.get('pert_mutilate_beta', 1.0))
    g2i = {g.upper(): i for i, g in enumerate(gene_names)}
    rng = np.random.default_rng(0)

    A_read = ad.read_h5ad(a.joung, backed='r')
    is_ctrl = (A_read.obs['is_control'].astype(str) == 'True').values
    tfg = A_read.obs['TF_gene'].astype(str).str.upper().values
    ctrl_rows = np.sort(rng.choice(np.where(is_ctrl)[0], min(a.n_ctrl, is_ctrl.sum()), replace=False))
    Xc = A_read.X[ctrl_rows]; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    eps_ctrl = model.encode(torch.from_numpy(Xc))[2]
    tfs = [t for t in list(dict.fromkeys(tfg[~is_ctrl])) if t in g2i][:a.n_tf]

    D0 = model.decoder.D.data.clone()
    ng, d = D0.shape
    D_perm = D0[torch.as_tensor(rng.permutation(ng))]
    Q, _ = torch.linalg.qr(torch.randn(ng, d)); D_rand = Q / (Q.norm() + 1e-9) * D0.norm()

    def doop(gi_list, disabled=False, Dover=None):
        if Dover is not None:
            model.decoder.D.data = Dover
        model.dag.disabled = disabled
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        z_prev = model.dag.scm_forward(eps_ctrl, A)
        z_post, u = model.intervene_shift_mutilate(eps_ctrl, gi_list, A, z_prev,
                                                   u_mode=u_mode, shift_beta=beta)
        delta = (model.decoder(z_post) - model.decoder(z_prev)).mean(0).numpy()
        model.dag.disabled = False
        if Dover is not None:
            model.decoder.D.data = D0.clone()
        return delta

    reals, pa, pdp, pdr, psid, norms = [], [], [], [], [], []
    perm = rng.permutation(len(tfs))
    perm = np.where(perm == np.arange(len(tfs)), (perm + 1) % len(tfs), perm)   # derangement-ish
    for k, t in enumerate(tfs):
        gi = g2i[t]
        dr = doop([gi]); reals.append(dr); norms.append(float(np.linalg.norm(dr)))
        pa.append(1 - cos(dr, doop([gi], disabled=True)))
        pdp.append(1 - cos(dr, doop([gi], Dover=D_perm)))
        pdr.append(1 - cos(dr, doop([gi], Dover=D_rand)))
        gj = g2i[tfs[perm[k]]]
        psid.append(1 - cos(dr, doop([gj])))                                  # other TF's shift
    reals = np.stack(reals)
    # cross-TF diversity: mean 1−cos between distinct real predictions
    div = []
    for i in range(len(reals)):
        for j in range(i + 1, len(reals)):
            div.append(1 - cos(reals[i], reals[j]))

    m = lambda x: float(np.mean(x))
    print(f"=== COMPONENT PROBE | {os.path.basename(os.path.dirname(os.path.dirname(a.checkpoint)))} "
          f"| {len(tfs)} TFs | 1−cos(Δ_real, Δ_variant), higher = more load-bearing ===")
    print(f"  probe_A       (A≡0)              = {m(pa):.4f}")
    print(f"  probe_Dperm   (decoder rows perm)= {m(pdp):.4f}")
    print(f"  probe_Drand   (decoder random)   = {m(pdr):.4f}")
    print(f"  probe_shiftID (other TF's shift) = {m(psid):.4f}")
    print(f"  crossTF_div   (pred diversity)   = {m(div):.4f}   [0 = all TF preds identical]")
    print(f"  mean ‖Δ_real‖                    = {m(norms):.4f}")


if __name__ == '__main__':
    main()
