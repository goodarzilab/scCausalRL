"""Synthetic-SCM positive control for the R4 architecture.

Plants a known SCM with GENERALIZABLE TF-specific structure and asks whether the REAL exp20
model+loss recovers it: graph-recovery AUROC (learned A vs planted A*), HELD-OUT-gene disc_resid,
and drift-removed A_gap. If R4 recovers the planted graph and generalizes zero-shot here — where the
signal is real, sparse, and a linear function of the descriptor — then the method is sound and Joung
is data-limited; if it fails even here, the architecture/objective has a bug.

Generative model (matches the exp20 structural do-operator so the arch is well-specified):
  descriptor_g ~ N(0, I_k)                       per-TF "ESM" analog (the generalization bridge)
  S*_g = mask_g ⊙ (W* @ descriptor_g)            SPARSE latent shift, LINEAR in descriptor → zero-shot
  control cell: z = (I − A*ᵀ)⁻¹ ε,   ε~N(0,I)
  pert cell g : u = clamp(β|S*_g|); z = (I − diag(1−u)A*ᵀ)⁻¹(ε + S*_g)   [nonlinear: tanh Picard]
  x = z @ D*ᵀ + σ·noise
TFs are gene-level split into train/test (test descriptors unseen).

Variants: --scm linear|nonlinear  × --decoder learn|frozen(=D*).  Run all from synthetic_poc.sbatch.
"""
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.causal_vae import CausalVAE
from training.losses import perturbation_set_loss_shift_mutilate, reconstruction_loss, kl_divergence
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def gen(d=16, G=200, n_tf=120, k=24, n_ctrl=2000, cells_per_tf=80, nonlinear=False, seed=0, graph='acyclic'):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    # planted A*. 'acyclic' = strictly upper-triangular DAG (nilpotent). 'cyclic' = general sparse matrix
    # (feedback loops allowed) — this MATCHES the model's actual class: spectral penalty only, NO
    # acyclicity (dagma_lambda=0), so any A with spectral radius <1 is representable. Both clamped <0.8.
    A = torch.zeros(d, d)
    if graph == 'acyclic':
        for i in range(d):
            for j in range(i + 1, d):
                if rng.random() < 0.25:
                    A[i, j] = torch.randn(1, generator=g).item() * 0.6
    else:  # cyclic: dense-ish random off-diagonal, sparsified to ~25%
        A = torch.randn(d, d, generator=g) * 0.6
        A = A * (torch.rand(d, d, generator=g) < 0.25).float()
        A.fill_diagonal_(0.0)                                                   # no self-loops
    sn = torch.linalg.matrix_norm(A, ord=2).item()
    if sn > 0.8:
        A *= 0.8 / sn
    Dstar = torch.linalg.qr(torch.randn(G, d, generator=g))[0]                 # orthonormal decoder [G,d]
    desc = torch.randn(G, k, generator=g)                                       # per-GENE descriptor [G,k]; TFs=first n_tf
    Wstar = torch.randn(d, k, generator=g) * (1.0 / np.sqrt(k))                 # descriptor -> shift
    Sfull = desc[:n_tf] @ Wstar.T                                               # [n_tf, d]
    mask = torch.zeros(n_tf, d)                                                 # sparse: 2 target programs/TF
    for t in range(n_tf):
        mask[t, rng.choice(d, 2, replace=False)] = 1.0
    Sstar = (Sfull * mask) * 2.5                                                # planted sparse latent shift

    def solve(eps, S=None):
        if S is None:
            u = torch.zeros(d); et = eps
        else:
            u = (S.abs()).clamp(max=1.0); et = eps + S
        if nonlinear:
            z = et.clone()
            for _ in range(40):
                z = (1 - u) * torch.tanh(z @ A) + et
            return z
        M = torch.eye(d) - (1 - u).unsqueeze(0) * A.T if S is not None else torch.eye(d) - A.T
        return torch.linalg.solve(M, et.unsqueeze(-1)).squeeze(-1)

    # control cells
    Xc = torch.stack([solve(torch.randn(d, generator=g)) for _ in range(n_ctrl)]) @ Dstar.T
    Xc += 0.1 * torch.randn(Xc.shape, generator=g)
    # perturbed cells per TF
    Xp = {}; de = torch.zeros(n_tf, G)
    cmean = Xc.mean(0)
    for t in range(n_tf):
        zs = torch.stack([solve(torch.randn(d, generator=g), Sstar[t]) for _ in range(cells_per_tf)])
        xp = zs @ Dstar.T + 0.1 * torch.randn(cells_per_tf, G, generator=g)
        Xp[t] = xp; de[t] = xp.mean(0) - cmean
    tf_perm = rng.permutation(n_tf); n_tr = int(0.8 * n_tf)
    return dict(A=A, D=Dstar, desc=desc, Sstar=Sstar, Xc=Xc, Xp=Xp, de=de,
                train=list(tf_perm[:n_tr]), test=list(tf_perm[n_tr:]), d=d, G=G, k=k, nonlinear=nonlinear)


def build_sentences(syn, tfs, B, S, rng):
    """Sentence batch: [B,S,G] ctrl + pert, gene_idx [B,1]."""
    Xc = syn['Xc']; sel = rng.choice(len(tfs), B)
    xc = torch.stack([Xc[rng.choice(len(Xc), S)] for _ in range(B)])
    xp = torch.stack([syn['Xp'][tfs[i]][rng.choice(len(syn['Xp'][tfs[i]]), S)] for i in sel])
    gi = torch.tensor([[tfs[i]] for i in sel], dtype=torch.long)
    return xc, xp, gi


def disc_resid(pred, obs, drift):
    Pr, Or = pred - drift, obs - drift
    Pz = (Pr - Pr.mean(1, keepdims=True)); Pz /= (np.linalg.norm(Pz, axis=1, keepdims=True) + 1e-8)
    Oz = (Or - Or.mean(1, keepdims=True)); Oz /= (np.linalg.norm(Oz, axis=1, keepdims=True) + 1e-8)
    C = Pz @ Oz.T; diag = np.diag(C)[:, None]
    beaten = (diag > C).sum(1) / (len(pred) - 1)
    pear = np.nanmean([np.corrcoef(Pr[i], Or[i])[0, 1] for i in range(len(pred))])
    return float(beaten.mean()), float(pear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scm', default='linear', choices=['linear', 'nonlinear'])
    ap.add_argument('--decoder', default='learn', choices=['learn', 'frozen'])
    ap.add_argument('--epochs', type=int, default=400); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--l0', type=float, default=0.02)   # target-gate L0 weight (sweep to force sparsity)
    ap.add_argument('--graph', default='acyclic', choices=['acyclic','cyclic'])
    ap.add_argument('--hard_target', action='store_true')   # perturbation-blind: shift only at true site
    a = ap.parse_args()
    torch.manual_seed(a.seed); nonlin = a.scm == 'nonlinear'
    syn = gen(nonlinear=nonlin, seed=a.seed, graph=a.graph)
    d, G, k = syn['d'], syn['G'], syn['k']
    import numpy as _np
    _ev = _np.abs(_np.linalg.eigvals(syn['A'].numpy())); print(f"=== SYNTHETIC POC | scm={a.scm} dec={a.decoder} graph={a.graph} hard_target={a.hard_target} | A* spectral_radius={_ev.max():.3f} cyclic_edges_lowertri={int((syn['A'].tril(-1).abs()>1e-6).sum())} | d={d} G={G} train/test TF={len(syn['train'])}/{len(syn['test'])} ===", flush=True)

    model = CausalVAE(n_genes=G, latent_dim=d, use_shift_encoder=True, shift_label_dim=k,
                      shift_hidden=128, shift_n_layers=1, decoder_type='linear',
                      pert_intervene_mode='structural', use_target_gate=True, target_gate_bias_init=3.0,
                      scm_nonlinear=nonlin, n_picard=40)
    model.set_pert_labels(syn['desc'])                                          # planted descriptor = the bridge
    if a.hard_target:
        og = torch.zeros(G, d); og[:syn['Sstar'].shape[0]] = (syn['Sstar'].abs() > 1e-6).float()  # true support
        model.oracle_gate = og                                                  # perturbation-blind hard target
    if a.decoder == 'frozen':
        with torch.no_grad():
            model.decoder.D.copy_(syn['D'])
        model.decoder.D.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-3)
    rng = np.random.default_rng(a.seed)
    de_t = syn['de']; drift_tr = de_t[syn['train']].mean(0)                     # leak-free drift (train TFs)
    pb = torch.zeros(G, G); pb[:de_t.shape[0]] = de_t                            # pseudobulk target [G,G]; TFs=first rows
    Xc_t = syn['Xc']
    for ep in range(a.epochs):
        model.train(); dag_on = ep >= a.epochs // 4
        model.dag.theta.requires_grad_(dag_on)
        # recon on control cells
        xb = Xc_t[rng.choice(len(Xc_t), 256)]
        out = model(xb, bypass_scm=not dag_on)
        loss = reconstruction_loss(out['x_hat'], xb) + 0.01 * kl_divergence(out['mu'], out['log_var'])
        if dag_on:
            xc, xp, gi = build_sentences(syn, syn['train'], B=16, S=32, rng=rng)
            pl, _ = perturbation_set_loss_shift_mutilate(
                xc, xp, gene_idx=gi, batch_id=torch.zeros(16, dtype=torch.long), modality_id=None,
                model=model, n_dag_samples=1, mse_weight=1.0, mmd_weight=0.0, u_mode='shift', shift_beta=1.0,
                contrastive_weight=1.0, drift=drift_tr, residual_target=True,
                pseudobulk_targets=pb, project_shared=True, target_l0_weight=a.l0)
            loss = loss + 5.0 * pl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 100 == 0:
            print(f"  ep{ep} loss={loss.item():.3f}", flush=True)

    # ---- eval ----
    model.eval()
    with torch.no_grad():
        A_learn = model.dag.sample_A(n_samples=1).squeeze(0)
        Astar = syn['A']
        # graph recovery: |A_learn| ranks the true edges of |A*|
        true_edge = (Astar.abs() > 1e-6).flatten().numpy().astype(int)
        au = roc_auc_score(true_edge, A_learn.abs().flatten().numpy()) if true_edge.sum() else float('nan')
        # held-out prediction (zero-shot via descriptor)
        eps0 = model.encode(Xc_t[:400])[2]
        z_prev = model.dag.scm_forward(eps0, A_learn); ctrl_recon = model.decoder(z_prev).mean(0).numpy()
        def pred_split(tfs, Av):
            P = np.zeros((len(tfs), G))
            for i, t in enumerate(tfs):
                S = model.compute_shift(torch.tensor([t])).sum(0)
                u = (S.abs()).clamp(max=1.0)
                zp, _, _ = model.dag.scm_intervene_u(eps0, Av, eps0 + S.unsqueeze(0), u)
                P[i] = model.decoder(zp).mean(0).numpy() - ctrl_recon
            return P
        drift = de_t[syn['train']].mean(0).numpy()
        te = syn['test']; obs = de_t[te].numpy()
        P_real = pred_split(te, A_learn); P_zero = pred_split(te, torch.zeros_like(A_learn))
        dc_r, pe_r = disc_resid(P_real, obs, drift)
        dc_z, _ = disc_resid(P_zero, obs, drift)
        # drift-removed A_gap on held-out: spearman(|Δ|) real vs zero
        def sp(P):
            return float(np.nanmean([spearmanr(np.abs((P - drift)[i]), np.abs((obs - drift)[i])).correlation
                                     for i in range(len(P))]))
        agap = sp(P_real) - sp(P_zero)
        # probe_A: does A CHANGE the held-out output (1−cos(Δ_realA, Δ_zeroA)), matching the Joung probe
        def _c(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        probe_A = float(np.mean([1 - _c(P_real[i], P_zero[i]) for i in range(len(te))]))
        # gate openness on held-out
        gate = float(model.compute_gate_prob(torch.tensor(te)).mean())
    print(f"  RESULT scm={a.scm} dec={a.decoder} l0={a.l0}: graph_AUROC={au:.3f} | heldout disc_resid={dc_r:.3f} "
          f"(zeroA {dc_z:.3f}) pear_resid={pe_r:.3f} | A_gap(drift-rem)={agap:+.3f} | probe_A={probe_A:.3f} | "
          f"gate_open={gate:.3f}", flush=True)


if __name__ == '__main__':
    main()
