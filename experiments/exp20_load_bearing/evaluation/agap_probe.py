"""Cheap A_gap probe — the decisive load-bearing check for exp20 arms.

For a set of held-out TFs, run the do-operator prediction Δ twice on the SAME model:
  - with the real A, and
  - with A≡0 (`model.dag.disabled = True`, runtime toggle),
each with its own consistent baseline (z_prev = scm_forward(ε, A_used)).
Report A_gap = 1 − cos(Δ_A, Δ_{A=0}), overall and restricted to NON-target genes
(genes whose latent target program is gated closed). A_gap≈0 ⇒ A inert (leak persists);
A_gap large (esp. on non-target genes) ⇒ A is load-bearing.

Usage: python evaluation/agap_probe.py --checkpoint <ckpt> [--n_tf 24 --n_ctrl 500]
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse as sp, anndata as ad
from evaluation.run_shift_eval_norman import load_shift_model

PROC = '/home/sgoudarzi/scCausalRL/experiments/exp17_joung_tfatlas/data/joung_singles_proc.h5ad'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--joung', default=PROC)
    ap.add_argument('--n_tf', type=int, default=24)
    ap.add_argument('--n_ctrl', type=int, default=500)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model, gene_names, ck = load_shift_model(a.checkpoint, 'cpu')
    mode = getattr(model, 'pert_intervene_mode', 'legacy')
    gate = getattr(model, 'use_target_gate', False)
    g2i = {g.upper(): i for i, g in enumerate(gene_names)}

    # follow the checkpoint's proc (v2 arms live on the 4019-gene axis, not the default 9923 proc)
    _jp = ((ck.get('config') or {}).get('data', {}) or {}).get('joung_proc_path') or a.joung
    A_read = ad.read_h5ad(_jp, backed='r')
    is_ctrl = (A_read.obs['is_control'].astype(str) == 'True').values
    tfg = A_read.obs['TF_gene'].astype(str).str.upper().values
    rng = np.random.default_rng(0)
    ctrl_rows = np.where(is_ctrl)[0]
    ctrl_rows = np.sort(rng.choice(ctrl_rows, min(a.n_ctrl, len(ctrl_rows)), replace=False))
    Xc = A_read.X[ctrl_rows]; Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
    eps_ctrl = model.encode(torch.from_numpy(Xc))[2]                      # [B, d]

    tfs = [t for t in list(dict.fromkeys(tfg[~is_ctrl])) if t in g2i][: a.n_tf]
    u_mode = (ck.get('config', {}).get('phase1', {}) or {}).get('pert_mutilate_u_mode', 'shift')
    beta = float((ck.get('config', {}).get('phase1', {}) or {}).get('pert_mutilate_beta', 1.0))

    def dooop(gi, disabled):
        model.dag.disabled = disabled
        A = model.dag.sample_A(n_samples=1).squeeze(0)
        z_prev = model.dag.scm_forward(eps_ctrl, A)
        z_post, u = model.intervene_shift_mutilate(eps_ctrl, [gi], A, z_prev,
                                                   u_mode=u_mode, shift_beta=beta)
        d = (model.decoder(z_post) - model.decoder(z_prev)).mean(0)      # [G] gene-space Δ
        model.dag.disabled = False
        return d.numpy(), u.numpy()

    def cos(a_, b_):
        a_ = a_ - a_.mean(); b_ = b_ - b_.mean()
        return float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-9))

    gaps, gaps_nt, opens = [], [], []
    for t in tfs:
        gi = g2i[t]
        dA, uA = dooop(gi, False)
        d0, _ = dooop(gi, True)
        gaps.append(1 - cos(dA, d0))
        # non-target genes: those whose top decoder-loading program is NOT a target program
        gp = model.compute_gate_prob(torch.tensor([gi]))[0].numpy()       # [d] open-prob
        opens.append(float(gp.mean()))
        tgt_programs = set(np.where(gp > 0.5)[0].tolist())
        D = model.decoder.D.numpy()                                       # [G, d]
        gene_top = np.abs(D).argmax(1)                                    # program per gene
        nt = np.array([j for j in range(len(gene_names)) if gene_top[j] not in tgt_programs])
        if len(nt) > 10:
            gaps_nt.append(1 - cos(dA[nt], d0[nt]))
    print(f"=== A_gap probe | mode={mode} gate={gate} | {len(tfs)} TFs ===")
    print(f"  gate_open_frac (mean) = {np.mean(opens):.3f}  (1.0 = dense/no-gate)")
    print(f"  A_gap (all genes)      = {np.mean(gaps):.4f}   [0=A inert, >0=A load-bearing]")
    print(f"  A_gap (non-target)     = {np.mean(gaps_nt):.4f}" if gaps_nt else "  A_gap (non-target)   = n/a")


if __name__ == '__main__':
    main()
