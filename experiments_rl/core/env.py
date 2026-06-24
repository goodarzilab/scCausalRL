"""Continuous-latent, goal-conditioned environment with a GENE-SPACE penalized reward.

Shared by every RL method (sequential PPO, set PPO, distillation) and consistent with
the MPC cost. No Leiden clustering; continuous z.

State        s_t = [z_t ; z_target]              (latent, goal-conditioned — what the
                                                   policy reads)
Action       a gene SET applied at one step      (singleton for sequential PPO,
                                                   k-set for set PPO, empty = no-op)
Transition   shift-native mutilation do-operator under the eval-A point estimate; ε
             threaded via recover_epsilon_u (persisted intervention → genuine
             multi-step counterfactual).
Reward       potential Φ(z) = signed-Pearson(decode(z) − ctrl_recon, target observed
             DE) on the target's DEGs — the falsifiable GENE-SPACE objective (latent
             distance was gameable). Penalized over the hard-Gumbel DAG ensemble:
                 r = (γΦ(z') − Φ(z)) − α·std_N(γΦ_ens(z')) − λ_cost
             Mean under eval-A; pessimism = ensemble disagreement (also returned as the
             OOD-honesty signal). Entropy-for-exploration is the PPO entropy bonus.
"""
import numpy as np
import torch

from core.scoring import observed_de, torch_match


class CausalLatentEnv:
    def __init__(self, op, data, max_steps=2, gamma=0.99, alpha_pess=1.0,
                 lambda_cost=0.01, tol=None, n_dag_samples=16, reward_scale=1.0,
                 deg_thresh=0.05):
        self.op = op
        self.data = data
        self.device = op.device
        self.d = op.d
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.alpha = float(alpha_pess)
        self.lambda_cost = float(lambda_cost)
        self.tol = tol
        self.N = int(n_dag_samples)
        self.reward_scale = float(reward_scale)
        self.thresh = float(deg_thresh)
        self.A_eval = op.mean_A()                                # validated point estimate
        self.ctrl_recon = op.decode(data.z0.unsqueeze(0))[0]     # [n_genes] (torch)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def _phi(self, z_rows):
        """Gene-space potential Φ = signed-Pearson(decode(z)−ctrl_recon, target DE) on
        the target's DEGs. z_rows [K, d] -> [K] in [-1, 1]."""
        pred = self.op.decode(z_rows) - self.ctrl_recon          # [K, n_genes]
        return torch_match(pred, self._o_t, self._sel_idx)

    @torch.no_grad()
    def reset(self, target_key, start_eps=None, rng=None):
        """target_key: a single gene (str) or combo (tuple). start = control-mean ε."""
        rng = rng or self._rng
        self.z_target = self.data.target_for(target_key).to(self.device)   # latent (for obs)
        lfc, mask = observed_de(self.data, target_key)
        sel = mask & (np.abs(lfc) > self.thresh)
        self._sel_idx = torch.tensor(sel.nonzero()[0], dtype=torch.long, device=self.device)
        self._o_t = torch.tensor(lfc[sel], dtype=torch.float32, device=self.device)

        if start_eps is None:
            start_eps = self.data.eps_ctrl_mean
        self.eps = start_eps.unsqueeze(0).to(self.device)                  # [1, d]
        self.A_batch = self.op.sample_dags(self.N)                         # [N, d, d]
        self.z = self.op.scm_forward(self.eps, self.A_eval)                # [1, d]
        self.step_count = 0
        self.applied_genes = []
        return self._obs()

    def _obs(self):
        return torch.cat([self.z.squeeze(0), self.z_target], -1)           # [2d]

    @torch.no_grad()
    def step(self, gene_set):
        """gene_set: list[int] applied simultaneously this step ([] = no-op)."""
        gene_set = list(gene_set)
        phi_prev = float(self._phi(self.z)[0])
        z_next, _u, eps_next = self.op.apply(self.eps, gene_set, self.A_eval)   # [1, d]
        phi_next = float(self._phi(z_next)[0])
        r_shape = self.gamma * phi_next - phi_prev

        # epistemic uncertainty: same action under the ensemble from the current ε
        z_ens, _, _ = self.op.apply_ensemble(self.eps.repeat(self.N, 1), gene_set, self.A_batch)
        phi_ens = self._phi(z_ens)                                         # [N]
        r_std = float((self.gamma * phi_ens).std(unbiased=False))
        disagreement = r_std

        reward = self.reward_scale * (r_shape - self.alpha * r_std) - self.lambda_cost * (len(gene_set) > 0)
        reward = float(np.nan_to_num(np.clip(reward, -10.0, 10.0)))

        self.eps = eps_next
        self.z = z_next
        self.step_count += 1
        self.applied_genes.extend(gene_set)

        match = phi_next
        done = (self.step_count >= self.max_steps) or \
               (self.tol is not None and match > self.tol)
        info = {'match': match, 'disagreement': disagreement,
                'r_mean': r_shape, 'r_std': r_std, 'applied': list(self.applied_genes)}
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    @torch.no_grad()
    def terminal_match(self, target_key, gene_set, start_eps=None):
        """One-shot helper: apply gene_set once from control; return (match, z_post)."""
        self.reset(target_key, start_eps=start_eps)
        _, _, _, info = self.step(gene_set)
        return info['match'], self.z.squeeze(0)
