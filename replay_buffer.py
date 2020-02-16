import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):

        self.state = np.zeros((max_size, state_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.actions = np.zeros((max_size, action_dim))
        self.rewards = np.zeros((max_size, 1))
        self.masks = np.ones((max_size, 1))

        self._max_size = max_size
        self._size = 0
        self._step = 0

        self.device = device

    def __len__(self):
        return self._size

    def step(self):
        return self._step

    def add(self, state, next_state, action, reward, done):
        self.state[self._step] = state
        self.next_state[self._step] = next_state
        self.actions[self._step] = action
        self.rewards[self._step] = reward
        self.masks[self._step] = not done

        self._size = min(self._size + 1, self._max_size)
        self._step = (self._step + 1) % self._max_size

    def sample(self, batch_size):
        indices = np.random.randint(0, self._size, size=batch_size)

        state_batch = self.state[indices]
        next_state_batch = self.next_state[indices]
        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        masks_batch = self.masks[indices]

        return (
            torch.tensor(state_batch, dtype=torch.float).to(self.device),
            torch.tensor(next_state_batch, dtype=torch.float).to(self.device),
            torch.tensor(actions_batch, dtype=torch.float).to(self.device),
            torch.tensor(rewards_batch, dtype=torch.float).to(self.device),
            torch.tensor(masks_batch, dtype=torch.float).to(self.device)
        )

