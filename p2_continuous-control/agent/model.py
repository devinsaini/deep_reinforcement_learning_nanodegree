import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, device):
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.device = device
        #self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        self.std = torch.nn.Parameter(torch.ones(1, action_dim)) * action_std_init

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 2 * action_dim),
                        nn.Tanh()
                    )
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.std = torch.nn.Parameter(torch.ones(1, self.action_dim) * new_action_std)


    def forward(self, state, action=None):
        action_params = self.actor(state)
        action_mean = action_params[..., :self.action_dim]
        action_std = action_params[..., self.action_dim:] + 1.0   # add 1.0 to clamp the std to be positive
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = torch.squeeze(self.critic(state), -1)
        return action, action_logprob, state_value, dist.entropy(), action_std.mean().detach()
