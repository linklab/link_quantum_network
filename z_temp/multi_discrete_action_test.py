import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import torch
import torch.nn as nn


class MultiDiscretePolicy(nn.Module):
    def __init__(self, input_dim, action_space):
        super(MultiDiscretePolicy, self).__init__()
        self.action_space = action_space
        self.fc = nn.Linear(input_dim, sum(action_space.nvec))

    def forward(self, x):
        logits = self.fc(x)
        # Multi-Discrete logits을 분리
        split_logits = torch.split(logits, list(self.action_space.nvec), dim=-1)

        return split_logits


if __name__ == "__main__":    
    # Multi-Discrete 액션 공간 정의 (예: [0~4, 0~1, 0~5])
    action_space = MultiDiscrete([5, 2, 6])
    print(action_space)
    multi_discrete_policy = MultiDiscretePolicy(10, action_space)
    x = torch.randn(1, 10)
    print(multi_discrete_policy(x))
