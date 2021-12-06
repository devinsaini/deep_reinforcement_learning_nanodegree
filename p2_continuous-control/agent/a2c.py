import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from agent.model import ActorCritic


class A2C:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, device, action_std_init=0.1):
        self.action_std = action_std_init
        self.gamma = gamma
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.MseLoss = nn.MSELoss()


    def act(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, _, _, _ = self.policy(state)

        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy()


    def update(self, state, action, logprobs, rewards):
        # get differentiable policy and value
        _, logprobs, state_values, dist_entropy, mean_std = self.policy(state, action=action)
        
        # compute advantages
        advantages = rewards - state_values.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # calculate loss
        policy_loss = -(logprobs.sum(dim=-1) * advantages).mean()
        value_loss = self.MseLoss(state_values, rewards)
        entropy_loss = -dist_entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.001 * entropy_loss
        
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return loss, mean_std

