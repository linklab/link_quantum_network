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

# Define quantum processor
def create_quantum_processor(name):
    noise_model = DepolarNoiseModel(depolar_rate=1e-3, time_independent=True)
    processor = QuantumProcessor(name, num_positions=2, memory_noise_models=[noise_model, noise_model])
    return processor

# Create nodes with quantum processors
node_A = Node("A", qmemory=create_quantum_processor("Processor_A"))
node_B = Node("B", qmemory=create_quantum_processor("Processor_B"))
node_C = Node("C", qmemory=create_quantum_processor("Processor_C"))

# Create quantum channels between nodes
channel_AB = QuantumChannel("QuantumChannel_A->B", length=10, models={"delay_model": FibreDelayModel()})
channel_BC = QuantumChannel("QuantumChannel_B->C", length=10, models={"delay_model": FibreDelayModel()})

# Wrap channels in DirectConnection objects
connection_AB = DirectConnection(name="A->B", channel_AtoB=channel_AB, channel_BtoA=QuantumChannel("QuantumChannel_B->A", length=10, models={"delay_model": FibreDelayModel()}))
connection_BC = DirectConnection(name="B->C", channel_AtoB=channel_BC, channel_BtoA=QuantumChannel("QuantumChannel_C->B", length=10, models={"delay_model": FibreDelayModel()}))

# Create a network and add nodes and connections
network = Network("QuantumNetwork")
network.add_node(node_A)
network.add_node(node_B)
network.add_node(node_C)
network.add_connection(node_A, node_B, connection=connection_AB)
network.add_connection(node_B, node_C, connection=connection_BC)

# Define a quantum entanglement generation protocol
class EntanglementProtocol(NodeProtocol):
    def __init__(self, node, peer, qubit_position):
        super().__init__(node)
        self.peer = peer
        self.qubit_position = qubit_position

    def run(self):
        qubits = ns.qubits.create_qubits(2)
        self.node.qmemory.put(qubits[0], positions=[self.qubit_position])
        self.peer.qmemory.put(qubits[1], positions=[self.qubit_position])
        ns.qubits.operate(qubits[0], ns.H)
        program = self.create_epr_program()
        self.node.qmemory.execute_program(program, qubit_mapping=[self.qubit_position, (self.qubit_position + 1) % 2])
        yield self.await_program(self.node.qmemory, program)
        self.peer.qmemory.execute_program(program, qubit_mapping=[self.qubit_position, (self.qubit_position + 1) % 2])
        yield self.await_program(self.peer.qmemory, program)

    def create_epr_program(self):
        program = QuantumProgram(num_qubits=2)
        program.apply(INSTR_H, 0)
        program.apply(INSTR_CNOT, [0, 1])
        return program

class SwapProtocol(NodeProtocol):
    def __init__(self, node_a, node_b, node_c):
        super().__init__(node_a)
        self.node_b = node_b
        self.node_c = node_c

    def run(self):
        q1, = ns.qubits.create_qubits(1)
        q2, = ns.qubits.create_qubits(1)
        self.node.qmemory.put(q1, positions=[0])
        self.node_b.qmemory.put(q2, positions=[0])
        ns.qubits.operate(q1, ns.H)
        ns.qubits.operate([q1, q2], ns.CNOT)

        q3, = ns.qubits.create_qubits(1)
        q4, = ns.qubits.create_qubits(1)
        self.node_b.qmemory.put(q3, positions=[1])
        self.node_c.qmemory.put(q4, positions=[0])
        ns.qubits.operate(q3, ns.H)
        ns.qubits.operate([q3, q4], ns.CNOT)

        result = np.random.rand() < 0.5  # Random success/failure for swapping
        return 1 if result else 0

# Define the reinforcement learning environment
class QuantumEntanglementEnv(Env):
    def __init__(self):
        super(QuantumEntanglementEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        self.observation_space = spaces.Box(low=-1, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = [0, 0, 0, 0, 0, -1]
        self.steps = 0
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
