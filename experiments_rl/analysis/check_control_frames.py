"""Are the Joung and Briggs CONTROL latents separate under the joint WM's encoder?

The RL always starts rollouts from the Joung hESC-control latent (data.eps_ctrl_mean) and applies TF shifts,
then scores against Briggs-frame targets. If the two datasets' *control* (pluripotent hESC) cells encode to
DIFFERENT latent regions, that start is a frame mismatch for Briggs goals. This measures it directly:
encode Joung control + Briggs pluripotent control through the SAME WM encoder and compare the ε clouds.
"""
import os, sys, argparse
import numpy as np, torch, anndata as ad, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); sys.path.insert(0, RL)
from core.setup_stack import add_stack_args, build_stack

PLURI = ['POU5F1', 'NANOG', 'SOX2', 'LIN28A', 'DPPA4', 'ZFP42']
BH = "/large_storage/goodarzilab/shervin/CausalBioRL/Briggs_TF_2026/_decompressed/all_cells_final_v2.h5ad"


def lognorm(X):
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.float32)
    return np.log1p(X / (X.sum(1, keepdims=True) + 1e-9) * 1e4)


def main():
    ap = argparse.ArgumentParser(); add_stack_args(ap)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--plurip_pct', type=float, default=0.10)
    a = ap.parse_args(); dev = torch.device(a.device)
    model, meta, op, data = build_stack(a, dev, n_dag=8)
    A = op.mean_A()
    epsJ = data.eps_ctrl.to(dev)                                      # [N_J, d] Joung control ε (from is_control)
    # ---- Briggs pluripotent controls, projected to the Joung gene axis, encoded through the SAME WM ----
    B = ad.read_h5ad(BH); bg = list(map(str, B.var_names)); gn = list(map(str, data.gene_names)); g2i = {g: i for i, g in enumerate(gn)}
    shared = [g for g in bg if g in g2i]; sidx = np.array([g2i[g] for g in shared])
    Xln = lognorm(B[:, shared].X); pmk = [shared.index(g) for g in PLURI if g in shared]
    ps = Xln[:, pmk].mean(1); isc = ps >= np.quantile(ps, 1 - a.plurip_pct)
    Xb = np.zeros((int(isc.sum()), len(gn)), np.float32); Xb[:, sidx] = Xln[isc]
    with torch.no_grad():
        out = op.model.encoder(torch.tensor(Xb, device=dev)); epsB = (out[2] if isinstance(out, (tuple, list)) else out).detach()
    mJ, mB = epsJ.mean(0), epsB.mean(0)
    between = float((mJ - mB).norm())
    wJ = float((epsJ - mJ).norm(dim=1).mean()); wB = float((epsB - mB).norm(dim=1).mean())
    cos = float(torch.nn.functional.cosine_similarity(mJ, mB, dim=0))
    z0J = op.scm_forward(epsJ, A).mean(0); z0B = op.scm_forward(epsB, A).mean(0)
    rJ = op.decode(z0J.unsqueeze(0))[0]; rB = op.decode(z0B.unsqueeze(0))[0]
    gcorr = float(torch.corrcoef(torch.stack([rJ, rB]))[0, 1])
    axis = (mJ - mB) / ((mJ - mB).norm() + 1e-9); pJ = epsJ @ axis; pB = epsB @ axis
    dprime = float(abs(pJ.mean() - pB.mean()) / ((pJ.std() + pB.std()) / 2 + 1e-9))
    print("=" * 78)
    print(f"Joung control N={epsJ.shape[0]} | Briggs pluripotent control N={epsB.shape[0]}")
    print(f"ε-space:  ||mean_J − mean_B|| = {between:.3f}   within-J {wJ:.3f}   within-B {wB:.3f}   "
          f"between/within ratio = {between / ((wJ + wB) / 2 + 1e-9):.2f}")
    print(f"ε centroid cosine(J,B) = {cos:.3f}")
    print(f"1-D separation on the J−B axis (d′ = |Δμ|/pooled-σ) = {dprime:.2f}   (>2 ⇒ clearly separated clouds)")
    print(f"gene-space control-recon corr(decode z0_J, decode z0_B) = {gcorr:.3f}   (1.0 ⇒ same control state; <1 ⇒ different)")
    print("=" * 78)


if __name__ == '__main__':
    main()
