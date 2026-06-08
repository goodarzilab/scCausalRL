import numpy as np
import torch


class RolloutBuffer:
    """
    Stores rollout data for PPO.
    """
    def __init__(self):
        self.states = []       # list of Tensor[pca_dim]
        self.targets = []      # list of Tensor[pca_dim] (target state embedding)
        self.actions = []      # list of int
        self.rewards = []      # list of float
        self.dones = []        # list of bool
        self.log_probs = []    # list of float
        self.values = []       # list of float
        self.returns = None
        self.advantages = None

    def add(self, state, target, action, reward, done, log_prob, value):
        self.states.append(state)
        self.targets.append(target)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """
        Compute returns and GAE advantages.
        last_value: float -- V(s_{T+1}) for bootstrapping
        """
        T = len(self.rewards)
        returns = np.zeros(T)
        advantages = np.zeros(T)

        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])

        gae = 0.0
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * values[t+1] * (1 - dones[t+1]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t+1]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        self.returns = returns
        self.advantages = advantages

    def get_batches(self, batch_size):
        """Yield mini-batches of (states, targets, actions, log_probs, returns, advantages)."""
        T = len(self.rewards)
        indices = np.random.permutation(T)

        states = torch.stack(self.states)      # [T, pca_dim]
        targets = torch.stack(self.targets)    # [T, pca_dim]
        actions = torch.tensor(self.actions, dtype=torch.long)
        log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)
        advantages = torch.tensor(self.advantages, dtype=torch.float32)

        for start in range(0, T, batch_size):
            batch_idx = indices[start:start+batch_size]
            yield (
                states[batch_idx],
                targets[batch_idx],
                actions[batch_idx],
                log_probs[batch_idx],
                returns[batch_idx],
                advantages[batch_idx],
            )

    def clear(self):
        self.states = []
        self.targets = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = None
        self.advantages = None

    def __len__(self):
        return len(self.rewards)
