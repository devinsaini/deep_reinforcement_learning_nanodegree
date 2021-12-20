import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """Actor Critic PyTorch implementation."""

    def __init__(self, state_dim, action_dim, device):
        """Initialize parameters and build model.

        Args:
            state_dim (int): Dimension of each state
            action_dim (int): Dimension of each action
            device (pytorch.device): Device to run the model on
        """
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.device = device

        hidden_nodes = 128
        hidden_act = nn.Tanh()

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, 2 * action_dim),
                        nn.Tanh()
                    )
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, hidden_nodes),
                        hidden_act,
                        nn.Linear(hidden_nodes, 1)
                    )


    def forward(self, state, action=None):
        """Build a network that maps states to action values. The network predicts mean and standard deviation for each action.

        Args:
            state (array_like): State input
            action (array_like, optional): Optional actions for which log probabilities should be returned. Defaults to None.

        Returns:
            array_like: tuple of actions, log probabilities, state values, entropy and predicted standard deviations
        """
        action_params = self.actor(state)
        action_mean = action_params[..., :self.action_dim]
        action_std = (action_params[..., self.action_dim:] + 1.0 + 1e-8) / 2.0   # add 1.0 to clamp the std to be positive
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        state_value = torch.squeeze(self.critic(state), -1)
        return action, action_logprob, state_value, dist.entropy(), action_std
