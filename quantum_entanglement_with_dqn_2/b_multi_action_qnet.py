import os
import sys
import random
from torch import nn
import torch.nn.functional as F
import collections
import torch
import numpy as np

print("TORCH VERSION:", torch.__version__)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(CURRENT_PATH, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

MODEL_DIR = os.path.join(PROJECT_HOME, "quantum_entanglement_with_dqn_2", "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, n_features=4, n_multi_actions=None):
        super(QNet, self).__init__()
        if n_multi_actions is None:
            n_multi_actions = [2, 2, 2]
        self.n_features = n_features
        self.n_multi_actions = n_multi_actions
        self.fc1 = nn.Linear(n_features, 128)  # fully connected
        self.fc2 = nn.Linear(128, 128)

        # Multi-head output for each discrete action dimension
        self.last_fc = []
        for n in range(len(self.n_multi_actions)):
            self.last_fc.append(nn.Linear(128, n_multi_actions[n]))

        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            x = x.unsqueeze(0) if x.ndim == 1 else x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        q_values = []
        for n in range(len(self.n_multi_actions)):
            q_values.append(self.last_fc[n](x))

        q_values = torch.stack(q_values, dim=1)  # shape: [batch_size, num_actions, num_values]
        return q_values

    def get_action(self, obs, epsilon=0.1):
        q_values = self.forward(obs)

        actions = []
        for n in range(len(self.n_multi_actions)):
            if random.random() < epsilon:
                actions.append(random.randrange(0, self.n_multi_actions[n]))
            else:
                actions.append(torch.argmax(q_values[:, n], dim=-1).item())

        return actions  # List of actions for each action dimension


Transition = collections.namedtuple(
    typename='Transition',
    field_names=['observation', 'action', 'next_observation', 'reward', 'done']
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self):
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 4), (32, 4)

        actions = np.array(actions)  # shape: [32, 3]
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (32, 3) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
