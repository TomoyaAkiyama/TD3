import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, max_size=1e6):
        self.state = torch.zeros([max_size, state_dim], device=device)
        self.next_state = torch.zeros([max_size, state_dim], device=device)
        self.rewards = torch.zeros([max_size, 1], device=device)
        self.actions = torch.zeros([max_size, action_dim], device=device)
        self.masks = torch.ones([max_size, 1], device=device)

        self._max_size = max_size
        self._size = 0
        self._step = 0

        self.device = device

    def __len__(self):
        return self._size

    def step(self):
        return self._step

    def add(self, state, next_state, action, reward, done):

        self.state[self._step].copy_(torch.tensor(state, dtype=torch.float).to(self.device))
        self.next_state[self._step].copy_(torch.tensor(next_state, dtype=torch.float).to(self.device))
        self.actions[self._step].copy_(torch.tensor(action, dtype=torch.float).to(self.device))
        self.rewards[self._step].copy_(torch.tensor(reward, dtype=torch.float).to(self.device))
        self.masks[self._step].copy_(torch.tensor([not done], dtype=torch.float).to(self.device))

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
            state_batch,
            next_state_batch,
            actions_batch,
            rewards_batch,
            masks_batch
        )
