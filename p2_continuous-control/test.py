from unityagents import UnityEnvironment
import torch
import numpy as np
from agent.ppo import PPO, RolloutBuffer
from agent.a2c import A2C
from collections import deque

from agent.trainer import Trainer

# set device to cpu or cuda
device = torch.device('cpu')

env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Linux/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def play_episode(env, brain_name, agent, trainer=None, max_t=1000, trajectory_length=16):
    env_info = env.reset(train_mode=True if trainer else False)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    for t in range(max_t):
        actions, action_log_probs = agent.act(states)
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards                         # update the score (for each agent)

        if trainer:
            trainer.step(states, actions, action_log_probs, rewards, dones)

        states = next_states
        if np.any(dones):
            break
    return scores


def train_agent(agent, env, brain_name, trainer, num_episodes=1000, max_t=1000, trajectory_length=16):
    scores_history = []
    for i_episode in range(1, num_episodes+1):
        scores = play_episode(env, brain_name, agent, trainer=trainer, max_t=max_t, trajectory_length=trajectory_length)
        avg_score = np.mean(scores)
        scores_history.append(avg_score)
        print('episode: {}\tscore: {}'.format(i_episode, avg_score))

        #if i_episode % 5 == 0:
        agent.policy.std = agent.policy.std * 0.99

    return scores_history

agent = A2C(state_size, action_size, 1e-3, 1e-3, 0.99, device, action_std_init=0.1)
#agent = PPO(state_size, action_size, 1e-3, 1e-3, 0.98, K_epochs=4, eps_clip=0.1, action_std_init=0.001)
trainer = Trainer(agent, device=device)

# train the agent
train_agent(agent, env, brain_name, trainer, num_episodes=50, max_t=2000, trajectory_length=100)

# validate
play_episode(env, brain_name, agent, trainer=None, max_t=2000)

env.close()