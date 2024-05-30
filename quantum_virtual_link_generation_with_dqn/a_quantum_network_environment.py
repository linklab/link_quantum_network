import gymnasium as gym
from gymnasium import spaces
import numpy as np
import netsquid as ns
from netsquid.qubits import ketstates as ks
from quantum_virtual_link_generation_with_dqn.utils.a_netsquid_based_quantum_network_generation import QuantumNetwork
from quantum_virtual_link_generation_with_dqn.utils.b_netsquid_based_protocols import EntanglementProtocol, \
    BellMeasurementProtocol


class QuantumNetworkEnv(gym.Env):
    def __init__(
        self,
        max_steps=5_000, light_v=200_000, initial_efficiency=1.0, mims_factor=2.0
    ):
        """
        ----------
          config
        ------------------------------------------------------------------------------
        env_name
        max_steps: maximum simulation steps
        light_v (km/s): light propagation speed in the fiber, km/s
        initial_efficiency: initial memory efficiency (Paper: a zero-time efficiency of 1.)
        mims_factor: memory decay coefficient
        ------------------------------------------------------------------------------
        """
        super(QuantumNetworkEnv, self).__init__()

        self.env_name = "QuantumNetwork"
        self.max_steps = max_steps

        self.light_v = light_v

        # For prob_s
        self.initial_efficiency = initial_efficiency
        self.mims_factor = mims_factor

        print("#" * 100)
        print(f"max_step: {self.max_steps:,}\t\t\tligt_v: {light_v:,}km/s")
        print(f"initial_efficiency: {self.initial_efficiency}")
        print(f"mims_factor: {self.mims_factor}")
        print(f"prob_s: {self.calculate_swap_success_probability([0.0, 0.0]):.2f} (attenuation)")

        # generate network & set protocols
        self.network = self.generate_network()
        self.set_protocols()
        self.fiber_lengths = self.network.quantum_channel_config["fiber_lengths"]
        print(f"fiber_length_1: {self.fiber_lengths[0]}km,\t\tfiber_length_2: {self.fiber_lengths[1]}km")
        # Slot Duration
        self.slot_duration = max([l / self.light_v for l in self.fiber_lengths])  # time step 단위
        print(f"slot_duration: {self.slot_duration}s")
        print("#" * 100)
        print()

        # define action space & observation space
        self.action_space = spaces.MultiDiscrete([2, 2, 2])  # actions: reset or wait for each link
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 2), dtype=np.float32)  # state: [entanglement status, age]

        self.reset()

    def generate_network(self):
        ns.set_qstate_formalism(ns.QFormalism.DM)
        network = QuantumNetwork()
        print(f"nodes: {network.node_lst}")
        print(f"quantum channels: {network.qchannel_lst}")
        return network

    def set_protocols(self):
        self.node_lst = self.network.node_lst
        self.qchannel_lst = self.network.qchannel_lst
        self.protocol_eba = EntanglementProtocol(self.node_lst[1], self.qchannel_lst[0], position=0)
        self.protocol_ebc = EntanglementProtocol(self.node_lst[1], self.qchannel_lst[1], position=1)
        self.protocol_swap = BellMeasurementProtocol(self.node_lst[1])

    def get_protocols(self, channel_num):
        if channel_num == 0:
            self.node_lst[0].qmemory.pop(positions=0)
            self.node_lst[1].qmemory.pop(positions=0)
            return self.protocol_eba
        elif channel_num == 1:
            self.node_lst[1].qmemory.pop(positions=1)
            self.node_lst[2].qmemory.pop(positions=0)
            return self.protocol_ebc
        elif channel_num == 2:
            return self.protocol_swap
        else:
            raise Exception("channel_num is should be in [0, 1, 2]")

    def check_elementary_link_entangled(self, channel_num):
        is_entangled = False
        if channel_num == 0:
            # delete qbits in qmemories
            q0, = self.node_lst[0].qmemory.peek(positions=[0])
            q1, = self.node_lst[1].qmemory.peek(positions=[0])
            if q0 is not None and q1 is not None:
                is_entangled = True
            # print(f"{q0 = }")
            # print(f"{q1 = }")
        elif channel_num == 1:
            # delete qbits in qmemories
            q2, = self.node_lst[1].qmemory.peek(positions=[1])
            q3, = self.node_lst[2].qmemory.peek(positions=[0])
            if q2 is not None and q3 is not None:
                is_entangled = True
            # print(f"{q2 = }")
            # print(f"{q3 = }")
        else:
            raise Exception("elementary channel_num is should be in [0, 1]")
        return is_entangled

    def get_fidelities(self):
        # node A
        q0, = self.node_lst[0].qmemory.peek(positions=[0])
        # node B
        q1, = self.node_lst[1].qmemory.peek(positions=[0])
        q2, = self.node_lst[1].qmemory.peek(positions=[1])
        # node C
        q3, = self.node_lst[2].qmemory.peek(positions=[0])
        link_1_fidelity = ns.qubits.fidelity([q0, q1], ks.b00)
        link_2_fidelity = ns.qubits.fidelity([q2, q3], ks.b00)
        return link_1_fidelity, link_2_fidelity

    def memory_efficiency(self, time):
        # Memory efficiency eta_m for Mims model
        effective_spin_coherence_time = 0.01   # 10 ms
        eta_m = self.initial_efficiency * np.exp(
            -2 * np.power(
                time / effective_spin_coherence_time,  # sec
                self.mims_factor
            )
        )
        return eta_m

    def calculate_swap_success_probability(self, times: list):
        time = max(times)
        ps = self.memory_efficiency(time)
        return ps

    def initialize_state(self, virtual_link_success=False):
        if virtual_link_success:
            self.state = np.array([[0, -1], [0, -1], [1, 0]], dtype=np.float32)
        else:
            self.state = np.array([[0, -1], [0, -1], [0, -1]], dtype=np.float32)
        # delete qbits in qmemories
        self.node_lst[0].qmemory.pop(positions=0)
        self.node_lst[1].qmemory.pop(positions=0)
        self.node_lst[1].qmemory.pop(positions=1)
        self.node_lst[2].qmemory.pop(positions=0)
        self.is_both_elementary_links_entangled = False

    def update_cutoff_time(self, link_num):
        if self.is_both_elementary_links_entangled:  # both links entangled
            self.cutoff_time_list.append(self.state[link_num][1])

    def reset(self):
        self.initialize_state()
        self.time_step = 0
        self.terminated = False
        self.truncated = False
        self.info = {}
        self.cutoff_time_list = []
        self.fidelities = []
        self.is_both_elementary_links_entangled = False
        self.number_of_successful_resets = [0, 0, 0]
        # reset simulation
        ns.sim_reset()
        return self.state.flatten(), self.info

    def step(self, action):
        self.time_step += 1
        reward = 0

        for i in range(2):  # for each elementary link
            if action[i] == 0:  # set or reset
                self.update_cutoff_time(i)
                # try to entangle
                entanglement_protocol = self.get_protocols(i)
                entanglement_protocol.reset()
                ns.sim_run()

                is_entangled = self.check_elementary_link_entangled(i)
                if is_entangled:
                    self.state[i] = [1, 0]   # successful entanglement
                    self.state[2] = [0, -1]  # reset virtual link
                    self.number_of_successful_resets[i] += 1
                else:
                    self.state[i] = [0, -1]  # entanglement failed
            else:  # wait
                if self.state[i][0] == 1:
                    assert self.state[i][1] != -1
                    self.state[i][1] += self.slot_duration  # increase age if entangled (sec)

        ns.sim_run(duration=self.slot_duration * 1e9)  # nanosec

        if self.state[0][0] == 1 and self.state[1][0] == 1:  # both links entangled
            self.is_both_elementary_links_entangled = True
        else:
            self.is_both_elementary_links_entangled = False

        if action[2] == 0:  # attempt swap set or reset
            if self.state[0][0] == 1 and self.state[1][0] == 1:  # both links entangled
                swap_success_prob = self.calculate_swap_success_probability(
                    [self.state[0][1], self.state[1][1]]
                )

                # check fidelities of elementary links
                link_1_fidelity, link_2_fidelity = self.get_fidelities()
                # print(f"{swap_success_prob = }")
                # print(f"{link_1_fidelity = }")
                # print(f"{link_2_fidelity = }")

                if link_1_fidelity <= 0.501 or link_2_fidelity <= 0.501:
                    swap_success_prob = 0.0

                if np.random.rand() < swap_success_prob:
                    self.get_protocols(2).reset()
                    self.initialize_state(virtual_link_success=True)  # consume entanglements of elementary links
                    self.number_of_successful_resets[2] += 1
                    self.fidelities.append([link_1_fidelity, link_2_fidelity])
                    reward = 1
                else:
                    self.state[2] = [0, -1]  # swap failed
        else:
            if self.state[2][0] == 1:   # virtual link entangled
                assert self.state[2][1] != -1
                self.state[2][1] += self.slot_duration     # increase age (sec)

        # check if done
        if self.time_step >= self.max_steps:
            self.terminated = True

        self.truncated = False
        self.info = {
            "number_of_successful_resets": self.number_of_successful_resets,
            "cutoff_time_list": self.cutoff_time_list,
            "fidelities": self.fidelities
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
