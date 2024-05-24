import gymnasium as gym
from gymnasium import spaces
import numpy as np


class QuantumNetworkEnv(gym.Env):
    def __init__(self):
        super(QuantumNetworkEnv, self).__init__()
        self.length = 10  # km
        self.attenuation_coefficient = 0.2  # dB/km
        self.lambda_decay = 0.01  # memory decay coefficient
        self.max_time = 1000  # maximum simulation steps

        self.action_space = spaces.MultiDiscrete([2, 2, 2])  # actions: reset or wait for each link
        self.observation_space = spaces.Box(low=0, high=self.max_time, shape=(3, 2), dtype=np.float32)  # state: [entanglement status, age]

        self.reset()

    def calculate_success_probability(self, length, attenuation_coefficient):
        initial_efficiency = 1.0
        return initial_efficiency * np.exp(-attenuation_coefficient * length)

    def memory_efficiency(self, time, lambda_decay):
        eta_m0 = 1.0
        return eta_m0 * np.exp(-lambda_decay * time)

    def calculate_swap_success_probability(self, time1, time2, lambda_decay, eta_m0=1.0):
        eta_m1 = self.memory_efficiency(time1, lambda_decay, eta_m0)
        eta_m2 = self.memory_efficiency(time2, lambda_decay, eta_m0)
        return min(eta_m1, eta_m2)

    def reset(self):
        self.state = np.array([[0, -1], [0, -1], [0, -1]], dtype=np.float32)
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.cutoff_time = [0, 0, 0]
        self.number_of_successful_resets = [0, 0, 0]
        return self.state.flatten(), self.info

    def step(self, action):
        self.time_step += 1
        reward = 0

        for i in range(2):  # for each elementary link
            if action[i] == 0:  # set or reset
                success_prob = self.calculate_success_probability(self.length, self.attenuation_coefficient)
                # print(f"{success_prob = }")
                if np.random.rand() < success_prob:
                    self.state[i] = [1, 0]  # entanglement successful
                    self.number_of_successful_resets[i] += 1
                else:
                    self.state[i] = [0, -1]  # entanglement failed
                    self.state[2] = [0, -1]
            else:  # wait
                if self.state[i][0] == 1:
                    assert self.state[i][1] != -1
                    self.state[i][1] += 1  # increase age if entangled

        if action[2] == 0:  # attempt swap set or reset
            if self.state[0][0] == 1 and self.state[1][0] == 1:  # both links entangled
                swap_success_prob = self.calculate_swap_success_probability(
                    self.state[0][1], self.state[1][1], self.lambda_decay
                )
                if np.random.rand() < swap_success_prob:
                    self.state[2] = [1, 0]  # virtual link successful
                    self.number_of_successful_resets[2] += 1
                    reward = 1
                else:
                    self.state[2] = [0, -1]  # virtual link failed
        else:
            # update the age of virtual link if exists
            if self.state[2][0] == 1:
                self.state[2][1] += 1

        # check if done
        if self.time_step >= self.max_time:
            self.terminated = True

        self.truncated = False
        self.info = {
            "number_of_successful_resets": self.number_of_successful_resets
        }

        return self.state.flatten(), reward, self.terminated, self.truncated, self.info


class RandomAgent(object):
    def __init__(self, env):
        self.env = env

    def get_action(self, obs):
        action = env.action_space.sample()
        return action


# 테스트 코드
if __name__ == "__main__":
    env = QuantumNetworkEnv()
    agent = RandomAgent(env)

    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(obs)  # 랜덤 행동 선택
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or terminated:
            done = True
        print(f"Time Step: {env.time_step}, Action: {action}, Observation: {obs}, Reward: {reward}, Total Reward: {total_reward}")
