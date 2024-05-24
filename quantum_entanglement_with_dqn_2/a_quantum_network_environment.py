import gymnasium as gym
from gymnasium import spaces
import numpy as np


class QuantumNetworkEnv(gym.Env):
    def __init__(self, config):
        """
        ----------
          config
        ------------------------------------------------------------------------------
        env_name
        max_steps: maximum simulation steps
        fiber_length: maximum simulation steps
        light_v: light propagation speed in the fiber, km/s
        attenuation_coefficient: fiber losses of 0.2 dB/km achievable around 1,550 nm
        lambda_decay: memory decay coefficient
        initial memory efficiency
        ------------------------------------------------------------------------------
        """
        super(QuantumNetworkEnv, self).__init__()

        self.env_name = config["env_name"]
        self.max_steps = config["max_steps"]
        self.fiber_length = config["fiber_length"]
        self.light_v = config["light_v"]
        self.attenuation_coefficient = config["attenuation_coefficient"] * np.log(10) / 10
        self.lambda_decay = config["lambda_decay"]
        self.eta0 = config["eta0"]
        self.slot_duration = self.fiber_length / self.light_v    # time step 단위

        self.action_space = spaces.MultiDiscrete([2, 2, 2])  # actions: reset or wait for each link
        self.observation_space = spaces.Box(low=0, high=self.max_steps, shape=(3, 2), dtype=np.float32)  # state: [entanglement status, age]

        self.reset()

    def calculate_entangle_success_probability(self):
        pe = np.exp(-self.attenuation_coefficient * self.fiber_length)
        return pe

    def memory_efficiency(self, t_max):
        # Memory efficiency eta_m for Mims model
        eta_m = self.eta0 * np.exp(-self.lambda_decay * t_max)
        return eta_m

    def calculate_swap_success_probability(self, times: list):
        time = max(times) * self.slot_duration
        ps = self.memory_efficiency(time)
        return ps

    def initialize_state(self, virtual_link_success=False):
        if virtual_link_success:
            self.state = np.array([[0, -1], [0, -1], [1, 0]], dtype=np.float32)
        else:
            self.state = np.array([[0, -1], [0, -1], [0, -1]], dtype=np.float32)
        self.is_both_elementary_links_entangled = False

    def update_cutoff_time(self, link_num):
        if self.is_both_elementary_links_entangled:  # both links entangled
            if link_num == np.argmax([self.state[0][1], self.state[1][1]]):
                self.cutoff_time_list.append(self.state[link_num][1])

    def reset(self):
        self.initialize_state()
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.cutoff_time_list = []
        self.is_both_elementary_links_entangled = False
        self.number_of_successful_resets = [0, 0, 0]
        return self.state.flatten(), self.info

    def step(self, action):
        self.time_step += 1
        reward = 0

        for i in range(2):  # for each elementary link
            if action[i] == 0:  # set or reset
                self.update_cutoff_time(i)
                success_prob = self.calculate_entangle_success_probability()
                # print(f"{i}:", f"{success_prob = }")
                if np.random.rand() < success_prob:
                    self.state[i] = [1, 0]  # entanglement successful
                    self.state[2] = [0, -1]  # reset virtual link
                    self.number_of_successful_resets[i] += 1
                else:
                    self.state[i] = [0, -1]  # entanglement failed
            else:  # wait
                if self.state[i][0] == 1:
                    assert self.state[i][1] != -1
                    self.state[i][1] += 1  # increase age if entangled

        if self.state[0][0] == 1 and self.state[1][0] == 1:  # both links entangled
            self.is_both_elementary_links_entangled = True
        else:
            self.is_both_elementary_links_entangled = False

        if action[2] == 0:  # attempt swap set or reset
            if self.is_both_elementary_links_entangled:  # both links entangled
                swap_success_prob = self.calculate_swap_success_probability(
                    [self.state[0][1], self.state[1][1]]
                )
                # print(f"{swap_success_prob = }", " | link_1's age:", f"{self.state[0][1]}", " | link_2's age:", f"{self.state[1][1]}")
                if np.random.rand() < swap_success_prob:
                    self.initialize_state(virtual_link_success=True)  # consume entanglements of elementary links
                    self.number_of_successful_resets[2] += 1
                    reward = 1
                else:
                    self.state[2] = [0, -1]  # swap failed
        else:
            if self.state[2][0] == 1:   # virtual link entangled
                assert self.state[2][1] != -1
                self.state[2][1] += 1     # increase age

        # check if done
        if self.time_step >= self.max_steps:
            self.terminated = True

        self.truncated = False
        self.info = {
            "number_of_successful_resets": self.number_of_successful_resets,
            "cutoff_time_list": self.cutoff_time_list
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
    env_config = {
        "env_name": "QuantumNetwork",
        "max_steps": 1000,                  # maximum simulation steps
        "fiber_length": 100,                # km
        "light_v": 200_000,                 # light propagation speed in the fiber, km/s
        "attenuation_coefficient": 0.2,     # dB/km
        "lambda_decay": 0.5,                # memory decay coefficient
        "eta0": 0.01                        # initial memory efficiency
    }
    
    env = QuantumNetworkEnv(env_config)
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
