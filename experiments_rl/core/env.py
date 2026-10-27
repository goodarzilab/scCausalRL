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

from core.scoring import observed_de, torch_match, torch_match_proj


class CausalLatentEnv:
    def __init__(self, op, data, max_steps=2, gamma=0.99, alpha_pess=1.0,
                 lambda_cost=0.01, tol=None, n_dag_samples=16, reward_scale=1.0,
                 deg_thresh=0.05, reward_mode='raw', drift=None, fate_reward=None, reward_clip=10.0,
                 maha_fn=None, beta_maha=0.0, maha_tau=0.0, reward_match='corr', cvar_reward_alpha=0.0):
        # cvar_reward_alpha>0: REPLACE the mean-γΦ − α·std pessimism with CVaR-lower over the N-DAG outcome
        # distribution — reward uses the mean of the WORST ceil(alpha*N) per-DAG matches (model-uncertainty
        # pessimism; no symmetry assumption). Needs n_dag_samples >= ~1/alpha for a stable tail. Maha stays outside.
        self.cvar_reward_alpha = float(cvar_reward_alpha)
        # reward_match: 'corr' = torch_match (direction-only Pearson, the default/original); 'proj' =
        # torch_match_proj (magnitude-aware projection). 'proj' rewards REACHING far toward the goal so the
        # policy does not collapse to a single well-aligned mover (the REPROcode high-k finding).
        self._match = torch_match_proj if reward_match == 'proj' else torch_match
        # maha_fn(z[K,d])->[K] support penalty (Joung-perturbation manifold); resid reward gets
        # -beta_maha*max(0, maha - maha_tau) (hinge => on-manifold proposals unpenalized).
        self.maha_fn = maha_fn; self.beta_maha = float(beta_maha); self.maha_tau = float(maha_tau)
        self.reward_clip = float(reward_clip)   # widen for fate mode: a hard [-10,10] clip flattens the
        # fate_reward (FateReward or None): if set, the env optimizes the Briggs HSPC fate objective
        # r~ = min-readout(z_post) - lam*DAG_std - gamma*Mahalanobis instead of the DE-match-to-target.
        # There is a single fixed goal (HSPC), so target_key is a dummy and z_target in the obs is 0.
        self.fate_reward = fate_reward
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
        # --- optional FRAME ROUTING: use a Briggs-frame control start+recon for Briggs/HSPC goals, Joung for Joung goals ---
        self._ctrl_recon_active = self.ctrl_recon                # swapped per-goal when frame routing is on
        self.eps0_active = data.eps_ctrl_mean                    # frame-appropriate start ε [d] (read by the H-step)
        self.z0_active = data.z0                                 # frame-appropriate control latent [d]
        self._frames = None; self._source_fn = None             # None => single Joung frame (default, unchanged)
        # reward_mode controls the potential Φ:
        #   'raw'   : signed-Pearson(Δ_pred, Δ_obs) on raw DEGs — drift-saturated (every TF
        #             aligns with the shared differentiation drift ⇒ Φ≈const ⇒ no gradient).
        #   'resid' : score the DRIFT-REMOVED Δ−drift — learnable signal, but optimizes the
        #             residual and the deployable metric is raw Δ (misaligned).
        #   'blend' : 0.5·Φ_raw + 0.5·Φ_resid — learnable (resid term) AND aligned to raw Δ.
        # drift = mean observed train Δ (the shared program).
        self.reward_mode = reward_mode
        self._drift = None
        if reward_mode in ('resid', 'blend'):
            if drift is None:
                drift = np.mean([data.de_single[g]['lfc']
                                 for g in data.single_split['train']], axis=0)
            self._drift = torch.tensor(np.asarray(drift, np.float32), device=self.device)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    def _phi(self, z_rows):
        """Gene-space potential Φ on the target's DEGs (raw / drift-removed / blend).
        z_rows [K, d] -> [K] in [-1, 1]."""
        pred = self.op.decode(z_rows) - self._ctrl_recon_active  # [K, n_genes] (frame-routed control recon)
        if self.reward_mode == 'raw':
            return self._match(pred, self._o_raw, self._sel_raw)
        if self.reward_mode == 'resid':
            return self._match(pred - self._drift, self._o_res, self._sel_res)
        # blend
        return 0.5 * self._match(pred, self._o_raw, self._sel_raw) \
            + 0.5 * self._match(pred - self._drift, self._o_res, self._sel_res)

    def set_frames(self, briggs_eps0, briggs_z0, briggs_ctrl_recon, source_fn):
        """Install FRAME ROUTING: goals whose source_fn(target_key) is 'briggs'/'hspc' plan from the Briggs-frame
        control (eps0, z0, ctrl_recon); 'joung' goals keep the default Joung frame. source_fn: target_key -> str."""
        self._frames = {'briggs': (briggs_eps0.to(self.device), briggs_z0.to(self.device), briggs_ctrl_recon.to(self.device))}
        self._source_fn = source_fn

    def set_kmax(self, default, per_source=None):
        """PER-SOURCE cocktail cap. default = global kmax; per_source = {'joung':k,'briggs':k,'hspc':k}.
        Routed by _source_fn (installed via set_frames); without it, kmax_active falls back to default.
        Motivation: the WM only saw Joung at k<=3, so uncapped k=6 Joung proposals are extrapolation."""
        self._kmax_default = int(default); self._kmax_map = dict(per_source or {}); self.kmax_active = int(default)

    def set_kmax_fate(self, fate_map):
        """PER-FATE cocktail cap keyed by fate name (overrides per-source in reset). Used by --mask_stop
        --force_k_mode answer to force each fate to exactly its answer-combo breadth."""
        self._kmax_fate_map = dict(fate_map or {})

    @torch.no_grad()
    def reset(self, target_key, start_eps=None, rng=None):
        """target_key: a single gene (str) or combo (tuple). start = control-mean ε."""
        rng = rng or self._rng
        if self.fate_reward is not None:                                   # fate mode: single fixed HSPC goal
            self.z_target = torch.zeros(self.d, device=self.device)
            self.eps = (self.data.eps_ctrl_mean if start_eps is None else start_eps).unsqueeze(0).to(self.device)
            self.A_batch = self.op.sample_dags(self.N)
            self.z = self.op.scm_forward(self.eps, self.A_eval)
            self.step_count = 0; self.applied_genes = []
            return self._obs()
        self.z_target = self.data.target_for(target_key).to(self.device)   # latent (for obs)
        lfc, mask = observed_de(self.data, target_key)
        sel = mask & (np.abs(lfc) > self.thresh)                 # raw DEGs
        self._sel_raw = torch.tensor(sel.nonzero()[0], dtype=torch.long, device=self.device)
        self._o_raw = torch.tensor(lfc[sel], dtype=torch.float32, device=self.device)
        if self._drift is not None:                              # drift-removed DEGs
            lfc_r = lfc - self._drift.cpu().numpy()
            sel_r = mask & (np.abs(lfc_r) > self.thresh)
            self._sel_res = torch.tensor(sel_r.nonzero()[0], dtype=torch.long, device=self.device)
            self._o_res = torch.tensor(lfc_r[sel_r], dtype=torch.float32, device=self.device)

        if self._frames is not None and self._source_fn is not None:       # FRAME ROUTING: pick control frame by goal source
            src = self._source_fn(target_key)
            if src in ('briggs', 'hspc'):
                e0, z0b, cr = self._frames['briggs']
                self.eps0_active = e0; self.z0_active = z0b; self._ctrl_recon_active = cr
                if start_eps is None: start_eps = e0
            else:
                self.eps0_active = self.data.eps_ctrl_mean; self.z0_active = self.data.z0; self._ctrl_recon_active = self.ctrl_recon
        if getattr(self, '_kmax_map', None) is not None:                   # PER-SOURCE cocktail cap (routed by _source_fn)
            src = self._source_fn(target_key) if self._source_fn is not None else None
            self.kmax_active = int(self._kmax_map.get(src, self._kmax_default))
        if getattr(self, '_kmax_fate_map', None):                          # PER-FATE cocktail cap (forced-answer-k) overrides per-source
            _nm = getattr(target_key, 'name', None)
            if _nm in self._kmax_fate_map: self.kmax_active = int(self._kmax_fate_map[_nm])
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
        if self.fate_reward is not None:                                   # fate objective (Briggs HSPC)
            if len(gene_set) == 0:
                return self._obs(), 0.0, True, {'match': 0.0, 'disagreement': 0.0, 'applied': []}
            r, base, u = self.fate_reward.pessimistic([gene_set])
            reward = float(np.nan_to_num(np.clip(float(r[0]) * self.reward_scale, -self.reward_clip, self.reward_clip)))
            z_next, _u, eps_next = self.op.apply(self.eps, gene_set, self.A_eval)
            self.eps = eps_next; self.z = z_next
            self.step_count += 1; self.applied_genes.extend(gene_set)
            info = {'match': float(base[0]), 'disagreement': float(u[0]), 'applied': list(self.applied_genes)}
            return self._obs(), reward, self.step_count >= self.max_steps, info
        phi_prev = float(self._phi(self.z)[0])
        z_next, _u, eps_next = self.op.apply(self.eps, gene_set, self.A_eval)   # [1, d]
        phi_next = float(self._phi(z_next)[0])
        r_shape = self.gamma * phi_next - phi_prev

        # epistemic uncertainty: same action under the ensemble from the current ε -> the N-DAG outcome distribution R_k
        z_ens, _, _ = self.op.apply_ensemble(self.eps.repeat(self.N, 1), gene_set, self.A_batch)
        phi_ens = self._phi(z_ens)                                         # [N] = {R_k}: Pearson_DEG(rv(S;A_k), cent_g)
        r_std = float((self.gamma * phi_ens).std(unbiased=False))
        disagreement = r_std

        maha_pen = 0.0
        if self.maha_fn is not None and self.beta_maha > 0 and len(gene_set) > 0:
            m = float(self.maha_fn(z_next)[0])
            maha_pen = self.beta_maha * max(0.0, m - self.maha_tau)          # hinge support penalty
        if self.cvar_reward_alpha > 0:                                      # CVaR-lower over the N-DAG outcomes (replaces mean − α·std; NOT stacked with it)
            kk = max(1, int(np.ceil(self.cvar_reward_alpha * phi_ens.numel())))
            cvar_low = float(torch.sort(phi_ens).values[:kk].mean())        # mean of the WORST ceil(alpha*N) per-DAG matches
            r_shape = self.gamma * cvar_low - phi_prev                      # pessimism is INSIDE the CVaR; no separate −α·std
            reward = self.reward_scale * r_shape - self.lambda_cost * (len(gene_set) > 0) - maha_pen
        else:
            reward = self.reward_scale * (r_shape - self.alpha * r_std) - self.lambda_cost * (len(gene_set) > 0) - maha_pen
        reward = float(np.nan_to_num(np.clip(reward, -self.reward_clip, self.reward_clip)))

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
