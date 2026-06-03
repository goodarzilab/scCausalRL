import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor_critic import Actor, Critic
from .replay import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization with GAE.
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
        n_epochs_per_update=4,
        batch_size=64,
        device='cpu',
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.n_epochs_per_update = n_epochs_per_update
        self.batch_size = batch_size
        self.device = device

        self.actor = Actor(state_dim=state_dim, n_actions=n_actions).to(device)
        self.critic = Critic(state_dim=state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.buffer = RolloutBuffer()

    def select_action(self, state_emb, target_emb):
        """
        Select action given state and target embeddings.
        Returns: (action, log_prob, value) as numpy scalars
        """
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            s = state_emb.unsqueeze(0).to(self.device)
            t = target_emb.unsqueeze(0).to(self.device)
            action, log_prob, entropy = self.actor.get_action(s, t)
            value = self.critic(s, t)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state_emb, target_emb, action, reward, done, log_prob, value):
        self.buffer.add(state_emb, target_emb, action, reward, done, log_prob, value)

    def update(self, last_state_emb=None, last_target_emb=None):
        """
        PPO update step. Called after collecting n_steps_per_rollout transitions.
        last_state_emb: embedding of terminal state for bootstrapping
        """
        # Get last value for bootstrapping
        if last_state_emb is not None:
            self.critic.eval()
            with torch.no_grad():
                s = last_state_emb.unsqueeze(0).to(self.device)
                t = last_target_emb.unsqueeze(0).to(self.device)
                last_value = self.critic(s, t).item()
        else:
            last_value = 0.0

        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        self.actor.train()
        self.critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs_per_update):
            for batch in self.buffer.get_batches(self.batch_size):
                states, targets, actions, old_log_probs, returns, advantages = batch

                states = states.to(self.device)
                targets = targets.to(self.device)
                actions = actions.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)

                # Normalize advantages (skip if batch has < 2 elements to avoid std() warning)
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                # NaN guard
                advantages = torch.nan_to_num(advantages, nan=0.0, posinf=10.0, neginf=-10.0)
                returns = torch.nan_to_num(returns, nan=0.0, posinf=10.0, neginf=-10.0)

                # Actor update
                new_log_probs, entropy = self.actor.evaluate_action(states, targets, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                actor_total = actor_loss + self.entropy_coeff * entropy_loss

                self.actor_optimizer.zero_grad()
                actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values = self.critic(states, targets)
                critic_loss = self.value_coeff * ((values - returns) ** 2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.buffer.clear()

        return {
            'actor_loss': total_actor_loss / max(n_updates, 1),
            'critic_loss': total_critic_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
