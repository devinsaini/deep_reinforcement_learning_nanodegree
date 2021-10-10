import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Trainer():
    """Manages the learning and replay memory for a reinforcement learning agent"""
    def __init__(self, agent, batch_size, update_every, min_samples, replay_buffer_size, device):
        """Initialize the trainer object

        Args:
            agent (Agent): Agent to train
            batch_size (int): batch size of samples usde in training
            update_every (int): execute learning update after this interval
            min_samples(int): minimum number of samples required in buffer to begin learning
            replay_buffer_size (int): maximum size of buffer
            device (Device): target pytorch device for creating model network
        """
        # Target agent
        self.agent = agent

        self.batch_size = batch_size
        self.update_every = update_every
        self.min_samples = min_samples

        # Replay memory
        self.memory = ReplayBuffer(agent.action_size, replay_buffer_size, batch_size, agent.seed, device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Adds the experience to replay memory. Also executes a learning update self.update_every steps.

        Args:
            state (array_like): observed state
            action (int): action taken in the given state
            reward (float): observed reward for taking given action in given state
            next_state (int): next observed state
            done (int): 1 for episode completion, 0 otherwise
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if(self.min_samples > 0):
            self.min_samples -= 1
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size and self.min_samples==0:
                experiences = self.memory.sample()
                self.agent.train(experiences)
