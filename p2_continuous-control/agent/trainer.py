import numpy as np
import torch

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def add(self, state, action, logprob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class Trainer:
    def __init__(self, agent, device) -> None:
        self.agent = agent
        self.rollout_buffer = RolloutBuffer()
        self.device = device

    def step(self, state, action, logprob, reward, done):
        self.rollout_buffer.add(state, action, logprob, reward, done)

        if np.any(done):
            # calculate discounted rewards
            disc_rewards = self.calc_disc_rewards(self.rollout_buffer.rewards)

            # create Tensors
            states = torch.FloatTensor(self.rollout_buffer.states).detach().to(self.device)
            actions = torch.FloatTensor(self.rollout_buffer.actions).detach().to(self.device)
            logprobs = torch.FloatTensor(self.rollout_buffer.logprobs).detach().to(self.device)
            disc_rewards = torch.FloatTensor(disc_rewards).detach().to(self.device)

            loss, mean_std = self.agent.update(states, actions, logprobs, disc_rewards)
            self.rollout_buffer.clear()

            print(f'Loss: {loss.item():.4f}, Mean Std: {mean_std:.4f}')

    def calc_disc_rewards(self, rewards):
        disc_rewards = []
        num_agents = torch.tensor(rewards).shape[1]
        discounted_reward = np.zeros(num_agents)
        for reward in reversed(rewards):
            discounted_reward = reward + (self.agent.gamma * discounted_reward)
            disc_rewards.insert(0, discounted_reward)
        return disc_rewards