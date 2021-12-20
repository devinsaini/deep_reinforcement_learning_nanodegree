import numpy as np

class RolloutBuffer:
    """Rollout buffer for training"""
    def __init__(self):
        """Initialize the rollout buffer"""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []

    def add(self, state, action, logprob, reward, state_value, done):
        """Add a new experience to the buffer

        Args:
            state (array_like): state values
            action (array_like): action values
            logprob (array_like): log probabilities
            reward (array_like): rewards
            state_value (array_like): state values
            done (array_like): done flags
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.state_values.append(state_value)
        self.dones.append(done)

    def clear(self):
        """Clear the buffer"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]



class TrajectoryBuffer:
    def __init__(self) -> None:
        self.trajectories = []
        self.current_trajectory = RolloutBuffer()
        self.max_length = 0

    def end_trajectory(self):
        self.max_length = max(self.max_length, len(self.current_trajectory.rewards))
        self.trajectories.append(self.current_trajectory)
        self.current_trajectory = RolloutBuffer()

    def clear(self):
        self.trajectories = []
        self.current_trajectory = RolloutBuffer()
        self.max_length = 0

    def combine_trajectories(self):
        states = self.combine_vector([t.states for t in self.trajectories])
        actions = self.combine_vector([t.actions for t in self.trajectories])
        logprobs = self.combine_vector([t.logprobs for t in self.trajectories])
        rewards = self.combine_vector([t.rewards for t in self.trajectories])
        state_values = self.combine_vector([t.state_values for t in self.trajectories])

        masks = self.combine_vector([t.dones for t in self.trajectories], fill_value=(-1, -1))
        masks = np.clip(masks, -1, 0) + 1

        return states, actions, logprobs, rewards, state_values, masks
    

    def combine_vector(self, vectors, fill_value=0):
        ar = np.array(vectors[0]).astype(float)
        combined = np.pad(ar, self.make_pad_spec(vectors[0]), 'constant', constant_values=fill_value)
        for v in vectors[1:]:
            ar = np.array(v).astype(float)
            padded = np.pad(ar, self.make_pad_spec(v), 'constant', constant_values=fill_value)
            combined = np.append(combined, padded, axis=1)
        return combined
    

    def make_pad_spec(self, vector):
            rank = len(np.array(vector).shape)
            if rank == 2:
                pad_spec = ((0, self.max_length - len(vector)), (0, 0))
            elif rank == 3:
                pad_spec = ((0, self.max_length - len(vector)), (0, 0), (0, 0))
            return pad_spec

