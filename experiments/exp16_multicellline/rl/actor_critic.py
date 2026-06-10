import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """
    Goal-conditioned policy: pi(a | z_t, z_target)

    Input: latent z-vector concatenated with target z-centroid
    Architecture: Linear(2*state_dim, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, n_actions)
    Output: Categorical distribution over discrete actions

    n_actions = n_genes * n_actions_per_gene (e.g. 2000 * 2 = 4000)
    """
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, s_emb, s_target_emb):
        """
        s_emb: Tensor[B, state_dim]
        s_target_emb: Tensor[B, state_dim]
        Returns: Categorical distribution
        """
        x = torch.cat([s_emb, s_target_emb], dim=-1)
        logits = self.net(x)
        return Categorical(logits=logits)

    def get_action(self, s_emb, s_target_emb):
        dist = self.forward(s_emb, s_target_emb)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_action(self, s_emb, s_target_emb, action):
        dist = self.forward(s_emb, s_target_emb)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


class Critic(nn.Module):
    """
    Goal-conditioned value function: V(z_t, z_target)

    Input: latent z-vector concatenated with target z-centroid
    Architecture: Linear(2*state_dim, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 1)
    Output: scalar value estimate
    """
    def __init__(self, state_dim):
        super().__init__()
        self.state_dim = state_dim

        self.net = nn.Sequential(
            nn.Linear(2 * state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s_emb, s_target_emb):
        """
        s_emb: Tensor[B, state_dim]
        s_target_emb: Tensor[B, state_dim]
        Returns: value Tensor[B]
        """
        x = torch.cat([s_emb, s_target_emb], dim=-1)
        return self.net(x).squeeze(-1)
