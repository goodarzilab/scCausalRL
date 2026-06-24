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
                 n_epochs=4, batch_size=64, device='cpu'):
        self.policy = SetPolicy(state_dim, n_candidates, k_max=k_max).to(device)
        self.critic = SetCritic(state_dim).to(device)
        self.opt_a = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.clip = clip_epsilon; self.ent_c = entropy_coeff; self.val_c = value_coeff
        self.max_grad_norm = max_grad_norm; self.n_epochs = n_epochs
        self.batch_size = batch_size; self.device = device
        self.k_max = k_max
        self._buf = []   # list of (s, tgt, tokens, reward, logp, value)

    @torch.no_grad()
    def select(self, s_emb, t_emb):
        self.policy.eval(); self.critic.eval()
        tokens, genes, logp, ent = self.policy.sample(s_emb, t_emb)
        v = float(self.critic(s_emb.unsqueeze(0), t_emb.unsqueeze(0))[0])
        return tokens, genes, logp, v

    def store(self, s_emb, t_emb, tokens, reward, logp, value):
        self._buf.append((s_emb.detach().cpu(), t_emb.detach().cpu(),
                          torch.tensor(tokens, dtype=torch.long), float(reward),
                          float(logp), float(value)))

    def update(self):
        if not self._buf:
            return {}
        S = torch.stack([b[0] for b in self._buf]).to(self.device)
        T = torch.stack([b[1] for b in self._buf]).to(self.device)
        TOK = torch.stack([b[2] for b in self._buf]).to(self.device)
        R = torch.tensor([b[3] for b in self._buf], device=self.device)
        OLDLP = torch.tensor([b[4] for b in self._buf], device=self.device)
        V = torch.tensor([b[5] for b in self._buf], device=self.device)
        adv = R - V                                              # 1-step advantage
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.nan_to_num(adv); ret = torch.nan_to_num(R)

        self.policy.train(); self.critic.train()
        n = S.shape[0]; stats = {}
        for _ in range(self.n_epochs):
            idx = torch.randperm(n, device=self.device)
            for st in range(0, n, self.batch_size):
                b = idx[st:st + self.batch_size]
                new_lp, ent = self.policy.evaluate(S[b], T[b], TOK[b])
                ratio = torch.exp(new_lp - OLDLP[b])
                surr1 = ratio * adv[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                actor_loss = -torch.min(surr1, surr2).mean() - self.ent_c * ent.mean()
                self.opt_a.zero_grad(); actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt_a.step()

                val = self.critic(S[b], T[b])
                critic_loss = self.val_c * ((val - ret[b]) ** 2).mean()
                self.opt_c.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_c.step()
                stats = {'actor_loss': float(actor_loss), 'critic_loss': float(critic_loss),
                         'entropy': float(ent.mean())}
        self._buf = []
        return stats

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'critic': self.critic.state_dict()}, path)
