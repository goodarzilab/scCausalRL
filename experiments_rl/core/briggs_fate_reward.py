"""Merged Briggs fate reward for RL:  r~(set) = Ridge_fate(z_post(set)) - lambda * u(set).

Track 2 (Ridge readout on the FROZEN WM latent) gives a trustworthy HSPC-fate score that HOLDS at high k
(Spearman ~0.5-0.7 vs raw one-step ~0.2). Track 1 (pessimism) penalizes epistemic uncertainty u so the
policy stays honest where even the readout extrapolates. Every planner (MPC / seq-PPO / set-PPO) scores a
candidate SET the same way: op.apply_sets(eps0, set, A) -> z_post -> Ridge readout, minus lambda * (DAG-
ensemble std of the fate score) [+ optional interaction-magnitude support term].

Readout npz = evaluation/briggs/readout_ridge_<tag>.npz (coef,intercept,mean,scale) from readout_head.py.
Use the LINEAR Ridge readout (generalizes to novel TFs); the MLP overfits and is intentionally not used here.
"""
import numpy as np
import torch


class FateReward:
    def set_support(self, feats_npz, support_gamma=0.0, exclude_tfs=None):
        """Load the OBSERVED-combo latent manifold (readout_feats Z [N,d]) and enable a Mahalanobis
        support penalty: u_support(z) = sqrt((z-μ)ᵀΣ⁻¹(z-μ)/d). This is the MOReL/MOPO fix for the
        distribution-shift exploitation — the readout is trustworthy ON the Briggs combo distribution,
        so penalize proposals that stray OFF it. support_gamma weights the penalty.
        exclude_tfs: drop support combos containing any of these TFs (leave-hit-out — removes leakage
        from the manifold so recovering held-out hits is a real generalization test)."""
        f = np.load(feats_npz, allow_pickle=True); Zall = f['Z']
        if exclude_tfs:
            ex = set(exclude_tfs); combo = f['combo']
            keep = np.array([not any(t in ex for t in c.split('|')) for c in combo])
            print(f"[support] leave-hit-out: kept {int(keep.sum())}/{len(combo)} combos (excluded {len(ex)} TFs)")
            Zall = Zall[keep]
        Z = torch.as_tensor(Zall, dtype=torch.float32, device=self.op.device)
        self.sup_mu = Z.mean(0)
        cov = torch.cov(Z.T) + 1e-3 * torch.eye(Z.shape[1], device=Z.device)
        self.sup_L = torch.linalg.cholesky(cov)                               # lower-triangular
        self.sup_d = Z.shape[1]; self.support_gamma = float(support_gamma)
        return self

    @torch.no_grad()
    def mahalanobis(self, z_post):
        """Mahalanobis distance of z_post [K,d] to the observed-combo manifold -> [K] (0 if no support set)."""
        if getattr(self, 'sup_L', None) is None:
            return torch.zeros(z_post.shape[0], device=z_post.device)
        diff = (z_post - self.sup_mu).T                                       # [d, K]
        sol = torch.linalg.solve_triangular(self.sup_L, diff, upper=False)    # [d, K]
        return ((sol * sol).sum(0) / self.sup_d).sqrt()                       # [K]

    def __init__(self, op, readout_npz, eps0, lam=1.0, n_dag=8, support_beta=0.0):
        """readout_npz: path OR list of paths. With >1 readout the fate score is the MIN across them
        (cross-readout pessimism: a cocktail must score high on EVERY axis, which suppresses axis-specific
        reward exploitation while keeping genuine HSPC cocktails, since the axes agree ~+0.81 on real ones)."""
        self.op = op
        dev, dt = op.device, torch.float32
        paths = readout_npz if isinstance(readout_npz, (list, tuple)) else [readout_npz]
        self.coef, self.intercept, self.mean, self.scale = [], [], [], []
        for p in paths:
            d = np.load(p)
            self.coef.append(torch.as_tensor(d['coef'], dtype=dt, device=dev))
            self.intercept.append(float(d['intercept']))
            self.mean.append(torch.as_tensor(d['mean'], dtype=dt, device=dev))
            self.scale.append(torch.as_tensor(d['scale'], dtype=dt, device=dev))
        self.n_readout = len(paths)
        self.eps0 = eps0.to(dev)                                              # control eps [d]
        self.lam = float(lam); self.n_dag = int(n_dag); self.support_beta = float(support_beta)
        self.A_eval = op.mean_A()
        self.has_head = getattr(op.model, 'interaction_U', None) is not None
        self.sup_L = None; self.support_gamma = 0.0

    def _fate_all(self, z_post):
        """z_post [K, d] -> per-readout fate [n_readout, K]."""
        return torch.stack([((z_post - self.mean[i]) / self.scale[i]) @ self.coef[i] + self.intercept[i]
                            for i in range(self.n_readout)])

    def _fate(self, z_post):
        """cross-readout-pessimistic fate score [K] = MIN over readouts."""
        return self._fate_all(z_post).min(dim=0).values

    @torch.no_grad()
    def per_readout(self, sets, A=None):
        """per-readout fate scores [n_readout, K] under the eval DAG (for reporting/exploitation checks)."""
        return self._fate_all(self.op.apply_sets(self.eps0, sets, self.A_eval if A is None else A))

    @torch.no_grad()
    def score(self, sets, A=None):
        """Point fate score for K gene SETS under the eval-mode DAG. Returns [K]."""
        return self._fate(self.op.apply_sets(self.eps0, sets, self.A_eval if A is None else A))

    @torch.no_grad()
    def interaction_norm(self, sets):
        """‖S_int(set)‖ — model-native OOD signal (grows as the pairwise head extrapolates past k=2)."""
        out = []
        for s in sets:
            if self.has_head and len(s) >= 2:
                gi = torch.as_tensor(list(s), dtype=torch.long, device=self.op.device)
                out.append(float(self.op.model.compute_interaction(gi).norm()))
            else:
                out.append(0.0)
        return torch.as_tensor(out, dtype=torch.float32, device=self.op.device)

    @torch.no_grad()
    def pessimistic(self, sets, A_batch=None):
        """r~ = fate - lambda * u.  u = DAG-ensemble std of the fate score (epistemic), + optional
        support_beta * ‖S_int‖. Returns (r~ [K], fate [K], u [K])."""
        base = self.score(sets)                                               # [K]
        if A_batch is None:
            A_batch = self.op.sample_dags(self.n_dag)
        z_eval = self.op.apply_sets(self.eps0, sets, self.A_eval)             # [K, d]
        fs = torch.stack([self._fate(self.op.apply_sets(self.eps0, sets, A_batch[n]))
                          for n in range(A_batch.shape[0])])                  # [N, K]
        u = fs.std(dim=0)                                                     # [K] epistemic (DAG)
        if self.support_beta > 0:
            u = u + self.support_beta * self.interaction_norm(sets)           # interaction-blowup OOD signal
        if self.support_gamma > 0:
            u = u + self.support_gamma * self.mahalanobis(z_eval)             # off-manifold OOD signal (MOReL/MOPO)
        return base - self.lam * u, base, u
