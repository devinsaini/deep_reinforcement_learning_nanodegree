import torch
import torch.nn as nn
from agent.model import ActorCritic


class PPO:
    """PPO agent"""
    def __init__(self, network, state_dim, action_dim, lr_actor, lr_critic, gamma, device, K_epochs, eps_clip):
        """Initialize the agent networks and parameters

        Args:
            network (torch.nn.module): Torch network to be used as policy
            state_dim (int): dimension of state space
            action_dim (int): dimension of action space
            lr_actor (float): learning rate for actor network
            lr_critic (float): learning rate for critic network
            gamma (float): discount factor
            device (torch.device): device to be used for training
            K_epochs (int): number of epochs for updating policy
            eps_clip (float): epsilon for clipping the PPO surrogate objective
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = network(state_dim, action_dim, device).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
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
            action, action_logprob, state_value, _, _  = self.policy(state)

        return action.detach().cpu().numpy(), action_logprob.detach().cpu().numpy(), state_value.detach().cpu().numpy()



    def update(self, state, action, old_logprobs, returns, advantages):
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

        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            _, logprobs, state_values, dist_entropy, mean_std = self.policy(state, action=action)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = (logprobs - old_logprobs).exp()
            surrogate = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)

            policy_loss = -(torch.min(ratios * advantages, surrogate * advantages)).mean()
            value_loss = self.MseLoss(state_values, returns)
            loss = policy_loss + value_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        return {
            'loss': loss.mean().item(),
            'entropy': dist_entropy.mean().item(),
            'std': mean_std.mean().item()
        }

    
    def save(self, checkpoint_path):
        """Save model parameters to file

        Args:
            checkpoint_path (string): file path to save the parameters
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        """Load model parameters from file

        Args:
            checkpoint_path (string): file path to load the parameters from
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        