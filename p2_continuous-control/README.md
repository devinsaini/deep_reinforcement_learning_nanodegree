# Project 2: Continuous Control

![Trained](images/trained.gif)

### Introduction

In this project, we will attempt to train an agent to control a double jointed robotic arm, moving its effector to a target location in 3D space.

A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.

The task is episodic, and in order to solve the environment, the agent must get an average score of +30.0 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below. We will be using the 20 agent version.  You need only select the environment that matches the operating system:
    - **Twenty (20) Agents version**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. Enter the path of the file in the `Continuous_Control.ipynb` notebook while initializing the environment.
   ```
   env = UnityEnvironment(file_name="Reacher.exe")
   ```

### Instructions
Follow the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

### Description
|File|Description|
|---|---|
|`Continuous_Control.ipynb`|main notebook for training and evaluation|
|`agent/a2c.py`|A2C agent implementation|
|`agent/ppo.py`|PPO agent implementation|
|`agent/model.py`|Actor Critic PyTorch network|
|`agent/trainer.py`|Training utility class|
|`checkpoints/ppo_checkpoint_159.pth`|trained PPO agent weights|
|`images/trained.gif`|animated trained agent visualization|
