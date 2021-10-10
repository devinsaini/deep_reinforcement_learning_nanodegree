import random

class Agent():
    """Base class for a RL agent"""

    def __init__(self, state_size, action_size, gamma, learning_rate, seed, device):
        """Initialize an agent

        Args:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma (float): discount factor
            learning_rate (float): optimizer learning rate
            seed (int): random seed
            device (Device): Pytorch target device
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.seed = random.seed(seed)
        self.device = device

    def compute_action(self, state, eps=0.):
        """Compute action to take in state

        Args:
            state (array_like): observed state
            eps (float, optional): exploration factor. Defaults to 0.

        Returns:
            int: action to take
        """
        return NotImplemented

    def train(self, experiences):
        """Execute training update

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        return NotImplemented