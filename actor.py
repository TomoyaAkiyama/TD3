import torch.nn as nn

activations = nn.ModuleDict([
    ['Tanh', nn.Tanh()],
    ['ReLU', nn.ReLU()]
])


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, action_max, activation='ReLU'):
        super(Actor, self).__init__()
        self.output_num = action_dim

        self.activation = activation

        layers = [
            nn.Linear(state_dim, hidden_size),
            activations[activation],
            nn.Linear(hidden_size, hidden_size),
            activations[activation],
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        ]
        self.actor = nn.Sequential(*layers)
        self.action_max = action_max

    def forward(self, state, noise=None):
        if noise is None:
            action = self.actor.forward(state)
            return self.action_max * action
        else:
            action = self.action_max * self.actor.forward(state) + noise
            return action.clamp(-self.action_max, self.action_max)
