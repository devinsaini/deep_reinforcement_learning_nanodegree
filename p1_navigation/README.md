[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, we will train a reinforcement learning agent to collect yellow bananas in a 3D environment.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches the operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. Enter the path of the file in the `Navigation.ipynb` notebook while initializing the environment.
   ```
   env = UnityEnvironment(file_name="Banana.exe")
   ```

### Instructions
Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### Description
|File|Description|
|---|---|
|`Navigation.ipynb`|main notebook for training and evaluation|
|`agents/agent.py`|base class for RL agents|
|`agents/dqn.py`|DQN agent implementation|
|`agents/double_dqn.py`|Double DQN agent implementation|
|`agents/utils.py`|Replay buffer and Trainer classes|
|`checkpoints/dqn.pth`|trained DQN agent weights|
|`checkpoints/ddqn.pth`|trained Double DQN agent weights|
