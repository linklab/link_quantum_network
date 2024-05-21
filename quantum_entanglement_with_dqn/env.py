import gymnasium as gym
from gymnasium import spaces
import numpy as np
import netsquid as ns
from netsquid.components import QuantumProcessor, QuantumChannel, ClassicalChannel
from netsquid.nodes import Node, Network, Connection
from netsquid.qubits import qubitapi as qapi
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits.qubitapi import create_qubits, operate
from netsquid.qubits.operators import H, CNOT
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class QuantumEntanglementEnv(gym.Env):
    def __init__(self):
        super(QuantumEntanglementEnv, self).__init__()
        self.network = self.create_network()
        self.node_a, self.node_b, self.node_c = self.create_nodes()
        self.qchannel_ab, self.qchannel_bc, self.cchannel_ab, self.cchannel_bc = self.create_channels()
        self.add_connections()
        self.action_space = spaces.Discrete(2)  # 0: WAIT, 1: RESET
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        self.max_steps = 1000
        self.current_step = 0
        self.reset()

    def create_quantum_processor(self, name):
        memory_noise_model = DepolarNoiseModel(depolar_rate=0.01, time_independent=True)
        processor = QuantumProcessor(name, num_positions=2, mem_noise_models=[memory_noise_model]*2)
        return processor

    def create_nodes(self):
        node_a = Node("A", qmemory=self.create_quantum_processor("A_processor"))
        node_b = Node("B", qmemory=self.create_quantum_processor("B_processor"))
        node_c = Node("C", qmemory=self.create_quantum_processor("C_processor"))
        self.network.add_nodes([node_a, node_b, node_c])
        return node_a, node_b, node_c

    def create_channels(self):
        qchannel_ab = QuantumChannel("qchannel_A2B", length=4)
        qchannel_bc = QuantumChannel("qchannel_B2C", length=4)
        cchannel_ab = ClassicalChannel("cchannel_A2B", length=4)
        cchannel_bc = ClassicalChannel("cchannel_B2C", length=4)
        return qchannel_ab, qchannel_bc, cchannel_ab, cchannel_bc

    def create_network(self):
        network = Network("Quantum Network")
        return network

    def add_connections(self):
        connection_ab = Connection(name="A_to_B")
        connection_bc = Connection(name="B_to_C")
        connection_ab.add_subcomponent(self.qchannel_ab, name="quantum")
        connection_ab.add_subcomponent(self.cchannel_ab, name="classical")
        connection_bc.add_subcomponent(self.qchannel_bc, name="quantum")
        connection_bc.add_subcomponent(self.cchannel_bc, name="classical")
        self.network.add_connection(self.node_a, self.node_b, connection=connection_ab)
        self.network.add_connection(self.node_b, self.node_c, connection=connection_bc)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.state = np.zeros(6, dtype=np.float32)
        self.done = False
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        if self.done:
            return self.state, 0, self.done, False, {}

        if action == 0:  # WAIT
            reward = 0
        elif action == 1:  # RESET
            reward = self.perform_entanglement_swapping()

        self.state = self.get_state()
        self.done = self.check_done()
        self.current_step += 1
        return self.state, reward, self.done, False, {}

    def perform_entanglement_swapping(self):
        q1, = create_qubits(1)
        q2, = create_qubits(1)
        self.node_a.qmemory.put(q1, positions=[0])
        self.node_b.qmemory.put(q2, positions=[0])
        operate(q1, H)  # Apply Hadamard gate to q1
        operate([q1, q2], CNOT)  # Apply CNOT gate

        q3, = create_qubits(1)
        q4, = create_qubits(1)
        self.node_b.qmemory.put(q3, positions=[1])
        self.node_c.qmemory.put(q4, positions=[0])
        operate(q3, H)  # Apply Hadamard gate to q3
        operate([q3, q4], CNOT)  # Apply CNOT gate

        result = np.random.rand() < 0.5  # Random success/failure for swapping
        return 1 if result else 0

    def get_state(self):
        return np.random.rand(6).astype(np.float32)  # Placeholder for actual state representation

    def check_done(self):
        return self.current_step >= self.max_steps  # End episode after max steps

    def render(self, mode='human'):
        print(f"State: {self.state}")