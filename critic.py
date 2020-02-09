import torch
import torch.nn as nn

activations = nn.ModuleDict([
    ['Tanh', nn.Tanh()],
    ['ReLU', nn.ReLU()]
])


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size, action_dim, activation='ReLU'):
        super(Critic, self).__init__()

        layers = [
            nn.Linear(state_dim + action_dim, hidden_size, bias=True),
            activations[activation],
            nn.Linear(hidden_size, hidden_size, bias=True),
            activations[activation],
            nn.Linear(hidden_size, 1, bias=True)
        ]
        self.Q1 = nn.Sequential(*layers)

        layers = [
            nn.Linear(state_dim + action_dim, hidden_size, bias=True),
            activations[activation],
            nn.Linear(hidden_size, hidden_size, bias=True),
            activations[activation],
            nn.Linear(hidden_size, 1, bias=True)
        ]
        self.Q2 = nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1.forward(state_action)
        q2 = self.Q2.forward(state_action)

        return q1, q2

    def q1(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        q1 = self.Q1.forward(state_action)

        return q1
