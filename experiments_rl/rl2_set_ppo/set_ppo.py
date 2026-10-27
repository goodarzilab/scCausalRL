"""M2 — PPO trainer for the set-based autoregressive policy.

Set-PPO episodes are ONE step: the policy emits a whole set, the env applies it at
once, and the gene-space reward scores the reached state. So GAE collapses to
advantage = return − V(s) (done after one step). The PPO core is otherwise standard;
the only change vs vanilla PPO is that the policy's log-prob and entropy are SUMS over
the autoregressive slots (SetPolicy.evaluate returns them), which PPO consumes as one
scalar per sample.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl2_set_ppo.set_policy import SetPolicy, SetCritic


class SetPPO:
    def __init__(self, state_dim, n_candidates, k_max=2, lr_actor=3e-4, lr_critic=1e-3,
                 clip_epsilon=0.2, entropy_coeff=0.02, value_coeff=0.5, max_grad_norm=0.5,
                 n_epochs=4, batch_size=64, device='cpu',
                 kl_bc_coeff=0.0, target_kl=None, warmup_iters=0, bc_policy=None, cvar_tau=0.0,
                 cvar_pergoal=False, rloo=False, cvar_anneal=0, cvar_loo=False, adv_norm_pergoal=False,
                 slot_head='mlp', mp_hops=0, mp_gate='gated', mp_op=None, decode_W=None):
        self.policy = SetPolicy(state_dim, n_candidates, k_max=k_max, slot_head=slot_head,
                                mp_hops=mp_hops, mp_gate=mp_gate, mp_op=mp_op, decode_W=decode_W).to(device)
        self.critic = SetCritic(state_dim).to(device)
        self.lr_actor = float(lr_actor)                          # peak actor LR (warmup ramps up to this)
        self.opt_a = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.clip = clip_epsilon; self.ent_c = entropy_coeff; self.val_c = value_coeff
        self.max_grad_norm = max_grad_norm; self.n_epochs = n_epochs
        self.batch_size = batch_size; self.device = device
        self.k_max = k_max
        self.cvar_tau = float(cvar_tau)   # >0 => RISK-SEEKING best-of-N: advantage vs a HIGH quantile of the batch reward (CVaR-upper) instead of the critic V
        self.cvar_pergoal = bool(cvar_pergoal)  # True => quantile computed WITHIN each goal's N samples (true per-goal best-of-N), not over the mixed batch
        self.rloo = bool(rloo)            # RLOO/GRPO: advantage = R_i - leave-one-out mean of its group (drops the critic baseline; for single-goal / G-samples-per-step)
        self.cvar_anneal = int(cvar_anneal)  # >0 => ramp advantage from R-V (critic) to CVaR over this many update() calls (warm-start so per-group CVaR can learn)
        self.cvar_loo = bool(cvar_loo)    # per-goal CVaR quantile computed LEAVE-ONE-OUT (over the OTHER G-1 samples), the unbiased GRPO-quantile baseline
        self.adv_norm_pergoal = bool(adv_norm_pergoal)  # normalize advantages WITHIN each goal group (PopArt-lite)
        # --- anti-BC-destruction knobs (all inert at their defaults => vanilla PPO) ---
        self.kl_bc_c = float(kl_bc_coeff)     # (1) beta*KL(pi_theta || pi_BC): anchor to the frozen BC policy
        self.target_kl = target_kl            # (2) break the epoch loop once approx-KL(pi_theta||pi_old) > this
        self.warmup_iters = int(warmup_iters) # (3) linear LR warmup 0 -> lr_actor over the first N update() calls
        self.bc_policy = bc_policy            # frozen SetPolicy snapshot taken right after BC-init
        self._iter = 0
        self._buf = []   # list of (s, tgt, tokens, reward, logp, value)

    @torch.no_grad()
    def select(self, s_emb, t_emb):
        self.policy.eval(); self.critic.eval()
        tokens, genes, logp, ent = self.policy.sample(s_emb, t_emb)
        v = float(self.critic(s_emb.unsqueeze(0), t_emb.unsqueeze(0))[0])
        return tokens, genes, logp, v

    def store(self, s_emb, t_emb, tokens, reward, logp, value, goal_id=-1):
        self._buf.append((s_emb.detach().cpu(), t_emb.detach().cpu(),
                          torch.tensor(tokens, dtype=torch.long), float(reward),
                          float(logp), float(value), int(goal_id)))

    def update(self):
        if not self._buf:
            return {}
        S = torch.stack([b[0] for b in self._buf]).to(self.device)
        T = torch.stack([b[1] for b in self._buf]).to(self.device)
        TOK = torch.stack([b[2] for b in self._buf]).to(self.device)
        R = torch.tensor([b[3] for b in self._buf], device=self.device)
        OLDLP = torch.tensor([b[4] for b in self._buf], device=self.device)
        V = torch.tensor([b[5] for b in self._buf], device=self.device)
        GID = torch.tensor([b[6] for b in self._buf], device=self.device)
        if self.rloo and R.numel() > 1:                          # RLOO / GRPO: advantage = R_i - (1/(G-1)) sum_{j!=i} R_j  (leave-one-out group mean, no critic)
            adv = R.clone()
            for g in torch.unique(GID):
                m = GID == g; Rg = R[m]; ng = int(m.sum())
                adv[m] = (Rg - (Rg.sum() - Rg) / (ng - 1)) if ng > 1 else Rg * 0.0
        elif self.cvar_tau > 0 and R.numel() > 1:                # CVaR: baseline = quantile(tau) => tau>0.5 risk-SEEKING (upper), tau<0.5 risk-AVERSE (lower)
            if self.cvar_pergoal:                                # per-goal quantile ("top tail = good for THIS goal"); LOO = over the other G-1
                cadv = R.clone()
                for g in torch.unique(GID):
                    idx = (GID == g).nonzero(as_tuple=True)[0]; Rg = R[idx]; ng = Rg.numel()
                    if ng <= 1:
                        cadv[idx] = 0.0
                    elif self.cvar_loo:                          # leave-one-out within-goal quantile (unbiased GRPO-quantile baseline)
                        for j in range(ng):
                            others = torch.cat([Rg[:j], Rg[j + 1:]])
                            cadv[idx[j]] = Rg[j] - torch.quantile(others, self.cvar_tau)
                    else:
                        cadv[idx] = Rg - torch.quantile(Rg, self.cvar_tau)
            else:                                                # batch-level quantile (keeps between-goal signal => learns cold, but easy-goal-biased)
                cadv = R - torch.quantile(R, self.cvar_tau)
            if self.cvar_anneal > 0:                             # ANNEAL: critic (R-V) early to break symmetry -> CVaR late once the policy conditions
                lam = min(1.0, self._iter / float(self.cvar_anneal))
                adv = (1.0 - lam) * (R - V) + lam * cadv
            else:
                adv = cadv
        else:
            adv = R - V                                          # 1-step advantage (standard critic baseline)
        if self.adv_norm_pergoal and adv.numel() > 1:              # per-goal normalization: low-range goals still contribute gradient
            for g in torch.unique(GID):
                m = GID == g
                if int(m.sum()) > 1:
                    adv[m] = (adv[m] - adv[m].mean()) / (adv[m].std() + 1e-8)
        elif adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.nan_to_num(adv); ret = torch.nan_to_num(R)
        # --- PPO diagnostics (on the collected batch, pre-update) ---
        ev = float(1.0 - (R - V).var() / (R.var() + 1e-8))       # explained variance of the critic
        rstd = float(R.std())                                     # reward std across the batch (gradient signal)
        # (3) linear LR warmup: ramp the actor LR 0 -> lr_actor over the first warmup_iters updates
        self._iter += 1
        lr_scale = 1.0 if self.warmup_iters <= 0 else min(1.0, self._iter / self.warmup_iters)
        for pg in self.opt_a.param_groups:
            pg['lr'] = self.lr_actor * lr_scale
        # (1) BC log-probs of the collected actions (frozen) — for the KL-to-BC anchor
        bc_lp = None
        if self.bc_policy is not None and self.kl_bc_c > 0:
            with torch.no_grad():
                bc_lp, _ = self.bc_policy.evaluate(S, T, TOK)

        self.policy.train(); self.critic.train()
        n = S.shape[0]; stats = {}; kls = []; clips = []; kl_bcs = []
        stopped = False
        for _ in range(self.n_epochs):
            if stopped:
                break
            idx = torch.randperm(n, device=self.device)
            for st in range(0, n, self.batch_size):
                b = idx[st:st + self.batch_size]
                new_lp, ent = self.policy.evaluate(S[b], T[b], TOK[b])
                ratio = torch.exp(new_lp - OLDLP[b])
                with torch.no_grad():                             # approx-KL + clip-fraction (PPO health)
                    logr = new_lp - OLDLP[b]
                    mb_kl = float(((ratio - 1) - logr).mean())              # SB3 approx-KL (>=0)
                    kls.append(mb_kl)
                    clips.append(float((ratio - 1).abs().gt(self.clip).float().mean()))
                # (2) target-KL early stop: bail the moment the policy moves too far from pi_old
                if self.target_kl is not None and mb_kl > self.target_kl:
                    stopped = True
                    break
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_c * ent.mean()
                if bc_lp is not None:                             # (1) + beta * KL(pi_theta || pi_BC), sampled on the batch actions
                    kl_bc = (new_lp - bc_lp[b]).mean()
                    kl_bcs.append(float(kl_bc.detach())); actor_loss = actor_loss + self.kl_bc_c * kl_bc
                self.opt_a.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_a.step()

                val = self.critic(S[b], T[b])
                critic_loss = self.val_c * ((val - ret[b]) ** 2).mean()
                self.opt_c.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_c.step()
                stats = {'actor_loss': float(actor_loss), 'critic_loss': float(critic_loss),
                         'entropy': float(ent.mean()), 'explained_variance': ev, 'reward_std': rstd,
                         'approx_kl': float(np.mean(kls)), 'clip_frac': float(np.mean(clips)),
                         'kl_bc': float(np.mean(kl_bcs)) if kl_bcs else 0.0,
                         'lr_actor': self.lr_actor * lr_scale}
        self._buf = []
        return stats

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'critic': self.critic.state_dict()}, path)
