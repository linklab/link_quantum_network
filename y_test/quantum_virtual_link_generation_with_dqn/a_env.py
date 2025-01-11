import netsquid as ns
from netsquid.nodes import Node, Network
from netsquid.components import QuantumChannel, QuantumProcessor
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel
from netsquid.protocols import NodeProtocol
from netsquid.nodes.connections import DirectConnection
from netsquid.components.qprogram import QuantumProgram
from gymnasium import Env, spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from netsquid.components.instructions import INSTR_H, INSTR_CNOT, INSTR_MEASURE


# Define the reinforcement learning environment
class QuantumVirtualLinkGenerationEnv(Env):
    def __init__(self):
        super(QuantumVirtualLinkGenerationEnv, self).__init__()
        # observation
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        self.step = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = [0, -1, 0, -1, 0, -1]
        self.step = 0
        info = {}
        return np.array(self.state, dtype=np.float32), info

    def step(self, action):
        done = False
        reward = 0
        ns.sim_reset()

        if action[0] == 1:  # Reset A-B
            protocol_AB = EntanglementProtocol(node_A, node_B, 0)
            protocol_AB.start()
            ns.sim_run()
            self.state[0] = 1 if node_A.qmemory.peek(0) is not None and node_B.qmemory.peek(0) is not None else 0
            self.state[1] = 0 if self.state[0] == 1 else self.state[1]

        if action[1] == 1:  # Reset B-C
            protocol_BC = EntanglementProtocol(node_B, node_C, 0)
            protocol_BC.start()
            ns.sim_run()
            self.state[2] = 1 if node_B.qmemory.peek(0) is not None and node_C.qmemory.peek(0) is not None else 0
            self.state[3] = 0 if self.state[2] == 1 else self.state[3]

        if action[2] == 1:  # Attempt entanglement swapping A-C
            if self.state[0] == 1 and self.state[2] == 1:
                swap_protocol = SwapProtocol(node_A, node_B, node_C)
                swap_protocol.start()
                ns.sim_run()
                result = swap_protocol.run()
                self.state[4] = 1 if result else 0
                self.state[5] = 0 if self.state[4] == 1 else self.state[5]

        for i in [1, 3, 5]:
            if self.state[i - 1] == 1:
                self.state[i] += 1

        if self.state[4] == 1:
            reward = 1
            self.state = [0, 0, 0, 0, 0, -1]

        self.steps += 1
        if self.steps >= 10000:
            done = True

        info = {}
        return np.array(self.state, dtype=np.float32), reward, done, False, info

    def render(self, mode='human', close=False):
        print(f"Step: {self.steps}, State: {self.state}, Reward: {reward}")

# Create and check the environment
env = QuantumEntanglementEnv()
check_env(env)

# Train the model
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Save the model
model.save("dqn_quantum_entanglement")

# Evaluate the model
obs, _ = env.reset()
episode_rewards = []
for i in range(10000):
    action, _states = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    episode_rewards.append(reward)
    if done:
        break

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title("Episode reward evolution")
plt.xlabel("Episodes")
plt.ylabel("Episode Rewards")
plt.legend(["Episode Rewards"])
plt.show()
