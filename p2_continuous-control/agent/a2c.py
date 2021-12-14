import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class A2C:
    """A2C agent with continuous action space"""
    def __init__(self, network, state_dim, action_dim, lr_actor, lr_critic, gamma, device):
        """Initialize A2C agent

        Args:
            network (torch.nn.module): Torch network to be used as policy
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            lr_actor (float): learning rate for actor network
            lr_critic (float): learning rate for critic network
            gamma (float): discount factor
            device (torch.device): device to be used for training
        """
        self.gamma = gamma
        self.device = device

        self.policy = network(state_dim, action_dim, device).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.MseLoss = nn.MSELoss()


    def act(self, state):
        """Returns actions for given state as per current policy

        Args:
            state (array_like): current state

        Returns:
            array_like, array_like, array_like: actions, log probabilities of actions and state values
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_value, _, _ = self.policy(state)

        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy(), state_value.detach().cpu().numpy()


    def update(self, state, action, logprobs, returns, advantages):
        """Update policy and value parameters using given batch of experience tuples

        Args:
            state (array_like): state values
            action (array_like): actions taken
            old_logprobs (array_like): log probabilities of actions
            returns (array_like): discounted future returns
            advantages (array_like): discounted advantages

        Returns:
            dict: loss and entropy of last update iteration
        """
        # get differentiable policy and value
        _, logprobs, state_values, dist_entropy = self.policy(state, action=action)

        # calculate loss
        policy_loss = -(logprobs * advantages).mean()
        value_loss = self.MseLoss(state_values, returns)
        entropy_loss = dist_entropy.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        return {
            'loss': loss.mean().item(),
            'entropy': dist_entropy.mean().item(),
        }