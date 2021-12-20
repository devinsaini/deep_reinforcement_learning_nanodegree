import numpy as np
import torch
from agent.buffers import TrajectoryBuffer

class Trainer:
    """Training helper"""
    def __init__(self, agent, device, trajectory_length, batch_size=32) -> None:
        """Initialize the trainer

        Args:
            agent (object): agent to train
            device (torch.device): device to train on
            trajectory_length (int): length of trajectory
            batch_size (int): batch size
        """
        self.agent = agent
        self.trajectory_buffer = TrajectoryBuffer()
        self.device = device
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size

        self.metrics = {
            'loss': [],
            'entropy': [],
            'std': []
        }


    def step(self, state, action, logprob, reward, state_values, done):
        """Add a new experience to the buffer. If batch is ready, update the agent.

        Args:
            state (array_like): state values
            action (array_like): action values
            logprob (array_like): log probabilities
            reward (array_like): rewards
            state_values (array_like): state values
            done (array_like): done flags
        """
        self.trajectory_buffer.current_trajectory.add(state, action, logprob, reward, state_values, done)

        if np.any(done):
            self.trajectory_buffer.end_trajectory()

        if len(self.trajectory_buffer.trajectories) == self.batch_size:
            states, actions, logprobs, rewards, state_values, masks = self.trajectory_buffer.combine_trajectories()
            self.update(states, actions, logprobs, rewards, state_values, masks)

        

    def update(self, states, actions, logprobs, rewards, state_values, masks):
        """Prepare batch and update the agent

        Args:
            states (array_like): state values
            actions (array_like): action values
            logprobs (array_like): log probabilities
            rewards (array_like): rewards
            state_values (array_like): state values
            masks (array_like): masks to avoid updating on done
        """

        # Calculate discounted returns
        returns = self.calc_returns(rewards)

        # create Tensors
        states = torch.FloatTensor(states).detach().to(self.device)
        actions = torch.FloatTensor(actions).detach().to(self.device)
        logprobs = torch.FloatTensor(logprobs).detach().to(self.device)
        returns = torch.FloatTensor(returns).detach().to(self.device)
        masks = torch.FloatTensor(masks).detach().to(self.device)

        # calculate advantage
        advantages = returns - torch.FloatTensor(state_values)
        advantages = advantages * masks     # mask the advantages for done trajectories to avoid updating the policy
        advantages = advantages.detach().to(self.device)

        # update policy
        train_info = self.agent.update(states, actions, logprobs, returns, advantages, masks)

        # update training metrics
        self.metrics['loss'].append(train_info['loss'])
        self.metrics['entropy'].append(train_info['entropy'])
        self.metrics['std'].append(train_info['std'])

        # clear rollout buffer
        self.trajectory_buffer.clear()


    def calc_returns(self, rewards):
        """Calculate discounted returns

        Args:
            rewards (array_like): rewards

        Returns:
            array_like: discounted returns
        """
        returns = np.zeros_like(rewards)
        returns_i = 0
        for i in reversed(range(len(rewards))):
            returns_i = rewards[i] + self.agent.gamma * returns_i
            returns[i] = returns_i
        return returns
