import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, n_features, action_space):
        super().__init__()
        self.n_features = n_features
        self.action_space = action_space
        self.fc1 = nn.Linear(n_features, 256)  # fully connected
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)        
        self.fc4 = nn.Linear(256, sum(self.action_space.nvec))
        self.to(DEVICE)

    def forward(self, x):
        if isinstance(x, np.ndarray) or isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))        
        logits = self.fc4(x)

        # print(logits, "!!!!!!!!!!!!!!!!")
        # Multi-Discrete logits을 분리
        multi_q_values = torch.split(logits, list(self.action_space.nvec), dim=-1)
        # print(split_logits, "!!!!")

        return multi_q_values

    def get_action(self, obs, epsilon):
        # random.random(): 0.0과 1.0사이의 임의의 값을 반환

        if random.random() < epsilon:
            actions = list(np.random.choice(2, 3))
        else:
            multi_q_values = self.forward(obs)
            actions = []
            for q_values in multi_q_values:
                actions.append(torch.argmax(q_values, dim=-1).item())

            # if epsilon == 0.0:
            #     print(obs, actions, "!!!!!!!!!!!!!!!! - 2")
            # else:
            #     print(obs, actions, "!!!!!!!!!!!!!!!! - 1")

        return actions  # argmax: 가장 큰 값에 대응되는 인덱스 반환
