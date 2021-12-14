import numpy as np
import torch

class RolloutBuffer:
    """Rollout buffer for training"""
    def __init__(self):
        """Initialize the rollout buffer"""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.next_state_values = []
        self.dones = []

    def add(self, state, action, logprob, reward, state_value, next_state_value, done):
        """Add a new experience to the buffer

        Args:
            state (array_like): state values
            action (array_like): action values
            logprob (array_like): log probabilities
            reward (array_like): rewards
            state_value (array_like): state values
            next_state_value (array_like): next state values
            done (array_like): done flags
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.next_state_values.append(next_state_value)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.next_state_values[:]
        del self.dones[:]


class Trainer:
    """Training helper"""
    def __init__(self, agent, device, trajectory_length, tau=None) -> None:
        """Initialize the trainer

        Args:
            agent (object): agent to train
            device (torch.device): device to train on
            trajectory_length (int): length of trajectory
            tau (float, optional): Generalized advantage estimation factor. Defaults to None.
        """
        self.agent = agent
        self.rollout_buffer = RolloutBuffer()
        self.device = device
        self.trajectory_length = trajectory_length
        self.tau = tau

        self.metrics = {
            'loss': [],
            'entropy': [],
            'std': []
        }


    def step(self, state, action, logprob, reward, state_values, next_state_values, done):
        """Add a new experience to the buffer. If batch is ready, update the agent.

        Args:
            state (array_like): state values
            action (array_like): action values
            logprob (array_like): log probabilities
            reward (array_like): rewards
            state_values (array_like): state values
            next_state_values (array_like): next state values
            done (array_like): done flags
        """
        self.rollout_buffer.add(state, action, logprob, reward, state_values, next_state_values, done)

        if len(self.rollout_buffer.states) == self.trajectory_length:
            self.update()
        elif np.any(done):
            # drop partial trajectories
            self.rollout_buffer.clear()
        

    def update(self):
        """Prepare the batch and update the agent"""

        # calculate discounted returns
        if self.tau:
            returns = self.calc_gae_returns(self.rollout_buffer.rewards, self.rollout_buffer.next_state_values, self.rollout_buffer.dones, self.tau)
        else:
            returns = self.calc_returns(self.rollout_buffer.rewards, self.rollout_buffer.next_state_values)

        # create Tensors
        states = torch.FloatTensor(self.rollout_buffer.states).detach().to(self.device)
        actions = torch.FloatTensor(self.rollout_buffer.actions).detach().to(self.device)
        logprobs = torch.FloatTensor(self.rollout_buffer.logprobs).detach().to(self.device)
        returns = torch.FloatTensor(returns).detach().to(self.device)

        # calculate advantage
        advantages = returns - torch.FloatTensor(self.rollout_buffer.state_values)
        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages = torch.FloatTensor(advantages).detach().to(self.device)

        # update policy
        train_info = self.agent.update(states, actions, logprobs, returns, advantages)

        # update training metrics
        self.metrics['loss'].append(train_info['loss'])
        self.metrics['entropy'].append(train_info['entropy'])
        self.metrics['std'].append(train_info['std'])

        # clear rollout buffer
        self.rollout_buffer.clear()


    def calc_returns(self, rewards, next_state_values):
        """Calculate discounted returns

        Args:
            rewards (array_like): rewards
            next_state_values (array_like): next state values

        Returns:
            array_like: discounted returns
        """
        returns = np.zeros_like(rewards)
        returns_i = next_state_values[-1]
        for i in reversed(range(len(rewards))):
            returns_i = rewards[i] + self.agent.gamma * returns_i
            returns[i] = returns_i
        return returns
        

    def calc_gae_returns(self, rewards, next_state_values, dones, tau):
        """Calculate discounted returns with Generalized Advantage Estimation

        Args:
            rewards (array_like): rewards
            next_state_values (array_like): next state values
            dones (array_like): done flags
            tau (float): Generalized advantage estimation factor

        Returns:
            array_like: discounted returns
        """
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.agent.gamma * next_state_values[i] * (1 - dones[i][0]) - self.rollout_buffer.state_values[i]
            gae = delta + self.agent.gamma * tau * (1 - dones[i][0]) * gae
            returns.insert(0, gae + self.rollout_buffer.state_values[i])
        return returns
