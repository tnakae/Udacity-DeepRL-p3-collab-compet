import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FCNetwork


class GaussianActorCriticNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hiddens=[64, 64]):
        super(GaussianActorCriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc_hidden = FCNetwork(state_dim, hiddens)
        self.fc_actor = nn.Linear(hiddens[-1], action_dim)
        self.fc_critic = nn.Linear(hiddens[-1], 1)
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def forward(self, states, actions=None):
        phi = self.fc_hidden(states)
        mu = F.tanh(self.fc_actor(phi))
        value = self.fc_critic(phi).squeeze(-1)

        dist = torch.distributions.Normal(mu, F.softplus(self.sigma))
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy, value

    def state_values(self, states):
        phi = self.fc_hidden(states)
        return self.fc_critic(phi).squeeze(-1)
