import networkx as nx
import matplotlib.pyplot as plt
import gymnasium as gym
import math
import random
import sys
from pathlib import Path

import numpy as np

BASE_PATH = str(Path(__file__).resolve().parent.parent)
assert BASE_PATH.endswith("link_quantum_network"), BASE_PATH
sys.path.append(BASE_PATH)

from b_agent.a_dummy.dummy_agent import DummyAgent


class Edge:
    def __init__(
            self, node0, node1, distance, entanglement=0, age=-1,
            last_entanglement_try_timestep=-1, last_entanglement_timestep=-1,
    ):
        self.node0 = node0
        self.node1 = node1
        self.distance = distance
        self.entanglement = entanglement
        self.age = age
        self.last_entanglement_try_timestep = last_entanglement_try_timestep
        self.last_entanglement_timestep = last_entanglement_timestep

        self.age_list = []
        self.average_age = 0.0
        self.entanglement_rate = 0.0

        self.cutoff_time_list = []
        self.average_cutoff_time = 0.0


class SimpleQuantumNetwork(nx.Graph):
    def __init__(self):
        super().__init__()

        # 노드 추가
        self.add_node("A")
        self.add_node("B")
        self.add_node("C")

        # 엣지 추가 (elementary links: e0, e1, virtual link: v)
        self.add_edge("A", "B", label="e0")  # elementary link e0
        self.add_edge("B", "C", label="e1")  # elementary link e1
        self.add_edge("A", "C", label="v", style="dashed")  # virtual link v

        self.edge_dict = {
            "e0": Edge(node0="A", node1="B", distance=0.1),
            "e1": Edge(node0="B", node1="C", distance=0.1),
            "v": Edge(node0="A", node1="C", distance=0.2)
        }  # distance unit: km

    def draw_graph(self):
        # 그래프 시각화
        pos = nx.spring_layout(self)  # 노드 배치 설정

        # 노드 그리기
        nx.draw_networkx_nodes(self, pos, node_size=500, node_color="lightblue")

        # 엣지 그리기
        nx.draw_networkx_edges(self, pos, edgelist=[("A", "B"), ("B", "C")], width=2)  # e0, e1
        nx.draw_networkx_edges(self, pos, edgelist=[("A", "C")], style="dashed", edge_color="gray", width=2)  # v

        # 노드 및 엣지 레이블 추가
        nx.draw_networkx_labels(self, pos, font_size=12, font_color="black")
        edge_labels = {("A", "B"): "e0", ("B", "C"): "e1", ("A", "C"): "v"}
        nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels, font_color="blue")

        # 그래프 출력
        plt.title("Elementary link vs. Virtual link")
        plt.show()


class SimpleQuantumNetworkEnv(gym.Env):
    optical_bsm_efficiency = 0.5
    probability_photon_detection = 0.2
    attenuation_length_of_the_optical_fiber = 22  # km

    probability_init_memory = 1.0

    t_0 = max_entanglement_age = 10

    running_average_window = 50

    def __init__(self, max_step):
        # s_t = [x_0, m_0, x_1, m_1, x_2, m_2]
        # x_k: {0, 1}, m_k: {-1, 0, ..., max_age}
        # x_k: 0 또는 1의 값을 가지는 이진 상태.
        # m_k: -1 (inactive) 또는 정수 값 (entanglement age).
        self.observation_space = gym.spaces.MultiDiscrete([
            2, self.max_entanglement_age + 2,  # (x_0, m_0)
            2, self.max_entanglement_age + 2,  # (x_1, m_1)
            2, self.max_entanglement_age + 2  # (x_2, m_2)
        ])

        # 각 링크(e0, e1, v)에 대해 {reset(0), wait(1)}의 선택 가능
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])

        # 현재 observation
        self.current_observation = None
        self.current_step = None

        self.quantum_network = SimpleQuantumNetwork()
        self.max_step = max_step

        self.probability_swap_list = []
        self.average_probability_swap = 0.0

    def get_observation(self):
        observation = [
            self.quantum_network.edge_dict["e0"].entanglement / self.max_entanglement_age,
            self.quantum_network.edge_dict["e0"].age / self.max_entanglement_age,
            self.quantum_network.edge_dict["e1"].entanglement / self.max_entanglement_age,
            self.quantum_network.edge_dict["e1"].age / self.max_entanglement_age,
            self.quantum_network.edge_dict["v"].entanglement / self.max_entanglement_age,
            self.quantum_network.edge_dict["v"].age / self.max_entanglement_age
        ]

        return observation

    def reset(self):
        self.quantum_network = SimpleQuantumNetwork()

        # 현재 observation 초기화
        self.current_observation = self.get_observation()
        self.current_step = 0
        self.probability_swap_list.clear()
        self.average_probability_swap = 0.0

        info = {
            "avg_swap_prob": self.average_probability_swap,
            "e0_avg_age": self.quantum_network.edge_dict["e0"].average_age,
            "e1_avg_age": self.quantum_network.edge_dict["e1"].average_age,
            "v_avg_age": self.quantum_network.edge_dict["v"].average_age,
            "e0_entanglement_rate": self.quantum_network.edge_dict["e0"].entanglement_rate,
            "e1_entanglement_rate": self.quantum_network.edge_dict["e1"].entanglement_rate,
            "v_entanglement_rate": self.quantum_network.edge_dict["v"].entanglement_rate,
            "e0_avg_cutoff_time": self.quantum_network.edge_dict["e0"].average_cutoff_time,
            "e1_avg_cutoff_time": self.quantum_network.edge_dict["e1"].average_cutoff_time,
            "v_avg_cutoff_time": self.quantum_network.edge_dict["v"].average_cutoff_time,
        }

        return self.current_observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.current_step += 1

        # action: [e0_action, e1_action, v_action]
        e0_action, e1_action, v_action = action

        ########################
        # e0 action processing #
        ########################
        if e0_action == 0:
            if self.quantum_network.edge_dict["e0"].entanglement == 1:
                self.aging_entanglement(edge_idx="e0")

            if self.quantum_network.edge_dict["e0"].age == self.max_entanglement_age:
                self.quantum_network.edge_dict["e0"].entanglement = 0
                self.quantum_network.edge_dict["e0"].age = -1

        elif e0_action == 1:
            self.try_entanglement(edge_idx="e0")
            self.update_cutoff_time(edge_idx="e0")
            self.quantum_network.edge_dict["e0"].last_entanglement_try_timestep = self.current_step

        else:
            raise ValueError()

        self.quantum_network.edge_dict["e0"].age_list.append(
            self.quantum_network.edge_dict["e0"].age
        )

        ########################
        # e1 action processing #
        ########################
        if e1_action == 0:
            if self.quantum_network.edge_dict["e1"].entanglement == 1:
                self.aging_entanglement(edge_idx="e1")

            if self.quantum_network.edge_dict["e1"].age == self.max_entanglement_age:
                self.quantum_network.edge_dict["e1"].entanglement = 0
                self.quantum_network.edge_dict["e1"].age = -1

        elif e1_action == 1:
            self.try_entanglement(edge_idx="e1")
            self.update_cutoff_time(edge_idx="e1")
            self.quantum_network.edge_dict["e1"].last_entanglement_try_timestep = self.current_step

        else:
            raise ValueError()

        self.quantum_network.edge_dict["e1"].age_list.append(
            self.quantum_network.edge_dict["e1"].age
        )
        #######################
        # v action processing #
        #######################
        reward = 0.0

        if v_action == 0:
            if self.quantum_network.edge_dict["v"].entanglement == 1:
                self.aging_entanglement(edge_idx="v")
                reward = 1

            if self.quantum_network.edge_dict["v"].age == self.max_entanglement_age:
                self.quantum_network.edge_dict["v"].entanglement = 0
                self.quantum_network.edge_dict["v"].age = -1

        elif v_action == 1:
            reward = self.try_swap()

            self.update_cutoff_time(edge_idx="v")
            self.quantum_network.edge_dict["v"].last_entanglement_try_timestep = self.current_step

        else:
            raise ValueError()

        self.quantum_network.edge_dict["v"].age_list.append(
            self.quantum_network.edge_dict["v"].age
        )

        # average_probability_swap 갱신
        self.probability_swap_list.append(self.probability_swap())
        # self.average_probability_swap = self.running_mean(self.probability_swap_list, self.running_average_window)
        self.average_probability_swap = np.mean(self.probability_swap_list)

        # 현재 observation 초기화
        self.current_observation = self.get_observation()
        next_observation = self.current_observation

        terminated = truncated = False
        if self.current_step == self.max_step:
            terminated = True

            filtered_age_list = [x for x in self.quantum_network.edge_dict["e0"].age_list if x != -1]
            self.quantum_network.edge_dict["e0"].average_age \
                = sum(filtered_age_list) / len(filtered_age_list) if len(filtered_age_list) != 0 else 0.0
            self.quantum_network.edge_dict["e0"].entanglement_rate \
                = len(filtered_age_list) / self.current_step

            filtered_age_list = [x for x in self.quantum_network.edge_dict["e1"].age_list if x != -1]
            self.quantum_network.edge_dict["e1"].average_age \
                = sum(filtered_age_list) / len(filtered_age_list) if len(filtered_age_list) != 0 else 0.0
            self.quantum_network.edge_dict["e1"].entanglement_rate \
                = len(filtered_age_list) / self.current_step

            filtered_age_list = [x for x in self.quantum_network.edge_dict["v"].age_list if x != -1]
            self.quantum_network.edge_dict["v"].average_age \
                = sum(filtered_age_list) / len(filtered_age_list) if len(filtered_age_list) != 0 else 0.0
            self.quantum_network.edge_dict["v"].entanglement_rate \
                = len(filtered_age_list) / self.current_step

        info = {
            "avg_swap_prob": self.average_probability_swap,
            "e0_avg_age": self.quantum_network.edge_dict["e0"].average_age,
            "e1_avg_age": self.quantum_network.edge_dict["e1"].average_age,
            "v_avg_age": self.quantum_network.edge_dict["v"].average_age,
            "e0_entanglement_rate": self.quantum_network.edge_dict["e0"].entanglement_rate,
            "e1_entanglement_rate": self.quantum_network.edge_dict["e1"].entanglement_rate,
            "v_entanglement_rate": self.quantum_network.edge_dict["v"].entanglement_rate,
            "e0_avg_cutoff_time": self.quantum_network.edge_dict["e0"].average_cutoff_time,
            "e1_avg_cutoff_time": self.quantum_network.edge_dict["e1"].average_cutoff_time,
            "v_avg_cutoff_time": self.quantum_network.edge_dict["v"].average_cutoff_time,
        }

        # max_age = max(
        #     self.quantum_network.edge_dict["e0"].age,
        #     self.quantum_network.edge_dict["e1"].age
        # )
        # reward -= max_age / self.max_step

        return next_observation, reward, terminated, truncated, info

    def try_entanglement(self, edge_idx):
        if random.random() <= self.probability_entanglement(edge_idx=edge_idx):
            self.quantum_network.edge_dict[edge_idx].entanglement = 1
            self.quantum_network.edge_dict[edge_idx].age = 0
        else:
            self.quantum_network.edge_dict[edge_idx].entanglement = 0
            self.quantum_network.edge_dict[edge_idx].age = -1

    def try_swap(self):
        reward = 0.0

        entanglement_conditions = [
            self.quantum_network.edge_dict["e0"].entanglement == 1,
            self.quantum_network.edge_dict["e1"].entanglement == 1
        ]
        if all(entanglement_conditions):
            if random.random() <= self.probability_swap():
                self.quantum_network.edge_dict["v"].entanglement = 1
                self.quantum_network.edge_dict["v"].age = 0
                reward = 1.0
            else:
                self.quantum_network.edge_dict["v"].entanglement = 0
                self.quantum_network.edge_dict["v"].age = -1

            self.quantum_network.edge_dict["e0"].entanglement = 0
            self.quantum_network.edge_dict["e0"].age = -1

            self.quantum_network.edge_dict["e1"].entanglement = 0
            self.quantum_network.edge_dict["e1"].age = -1
        else:
            pass

        return reward

    def update_cutoff_time(self, edge_idx):
        # cutoff_time_list 에 새로운 값 추가
        if self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep != -1:
            self.quantum_network.edge_dict[edge_idx].cutoff_time_list.append(
                self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep
            )
            # self.quantum_network.edge_dict[edge_idx].average_cutoff_time = self.running_mean(
            #     self.quantum_network.edge_dict[edge_idx].cutoff_time_list, self.running_average_window
            # )
            self.quantum_network.edge_dict[edge_idx].average_cutoff_time = np.mean(
                self.quantum_network.edge_dict[edge_idx].cutoff_time_list
            )

    def update_edge_age(self, edge_idx):
        if self.quantum_network.edge_dict[edge_idx].last_entanglement_timestep != -1:
            self.quantum_network.edge_dict[edge_idx].age_list.append(
                self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_timestep
            )
            # self.quantum_network.edge_dict[edge_idx].average_age = self.running_mean(
            #     self.quantum_network.edge_dict[edge_idx].age_list, self.running_average_window
            # )
            self.quantum_network.edge_dict[edge_idx].average_age = np.mean(
                self.quantum_network.edge_dict[edge_idx].age_list
            )

    # def update_edge_age(self, edge_idx):
    #     if self.quantum_network.edge_dict[edge_idx].age != -1:
    #         self.quantum_network.edge_dict[edge_idx].age_list.append(
    #             self.quantum_network.edge_dict[edge_idx].age + 1
    #         )
    #         # self.quantum_network.edge_dict[edge_idx].average_age = self.running_mean(
    #         #     self.quantum_network.edge_dict[edge_idx].age_list, self.running_average_window
    #         # )
    #         self.quantum_network.edge_dict[edge_idx].average_age = np.mean(
    #             self.quantum_network.edge_dict[edge_idx].age_list
    #         )

    def aging_entanglement(self, edge_idx):
        self.quantum_network.edge_dict[edge_idx].age += 1

    def probability_entanglement(self, edge_idx):
        edge_distance = self.quantum_network.edge_dict[edge_idx].distance

        prob_entanglement = (1.0 / 2.0) * self.optical_bsm_efficiency * \
                            (self.probability_photon_detection ** 2) * \
                            math.e ** (-1.0 * edge_distance / self.attenuation_length_of_the_optical_fiber)

        return prob_entanglement

    def probability_swap(self):
        entanglement_conditions = [
            self.quantum_network.edge_dict["e0"].entanglement == 1,
            self.quantum_network.edge_dict["e1"].entanglement == 1
        ]
        if all(entanglement_conditions):
            max_age = max(
                self.quantum_network.edge_dict["e0"].age,
                self.quantum_network.edge_dict["e1"].age
            )
            prob_swap = self.probability_valid_state(max_age)
        else:
            prob_swap = 0.0

        return prob_swap

    def probability_valid_state(self, age):
        prob_valid_entanglement = self.probability_init_memory * math.e ** (-1.0 * age / self.t_0)

        return prob_valid_entanglement

    def render(self):
        pass

    def running_mean(self, l, N):
        sum_ = 0
        result = list(0 for x in l)

        if len(l) < N:
            return sum(l) / len(l)
        else:
            for i in range(0, N):
                sum_ = sum_ + l[i]
                result[i] = sum_ / (i + 1)

            for i in range(N, len(l)):
                sum_ = sum_ - l[i - N] + l[i]
                result[i] = sum_ / N

            return result[-1]


def test_env():
    # SimpleQuantumNetwork 객체 생성
    simple_quantum_network = SimpleQuantumNetwork()
    # simple_quantum_network.draw_graph()

    # SimpleQuantumNetworkEnv 객체 생성
    env = SimpleQuantumNetworkEnv(max_step=100)
    print(env.observation_space)
    print(env.action_space)
    print(env.reset())
    print("probability_entanglement(0): {0}".format(env.probability_entanglement("e0")))
    print("probability_entanglement(1): {0}".format(env.probability_entanglement("e1")))
    print("probability_valid_state(0): {0}".format(env.probability_valid_state(age=0)))
    print("probability_valid_state(7): {0}".format(env.probability_valid_state(age=7)))
    print("probability_valid_state(10): {0}".format(env.probability_valid_state(age=10)))
    print(env.probability_swap())
    print(env.try_entanglement(0))
    print()


def loop_test():
    env = SimpleQuantumNetworkEnv(max_step=1000)
    observation, info = env.reset()

    dummy_agent = DummyAgent()

    episode_step = 0
    episode_reward = 0.0
    done = False

    while not done:
        action = dummy_agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action=action)

        done = terminated or truncated
        episode_reward += reward
        episode_step += 1

        print(
            "[Step: {0:>3}] Ob.: {1:<22} Act: {2} N.Ob.: {3:<22} "
            "R.: {4} Term.: {5:<5}".format(
                episode_step,
                str(observation),
                action,
                str(next_observation),
                reward,
                str(terminated),
            ),
            end=" "
        )
        print(
            "S.P.: {:5.3f}".format(info["avg_swap_prob"]),
            "age: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                info["e0_avg_age"], info["e1_avg_age"], info["v_avg_age"]
            ),
            "cutoff_time: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                info["e0_avg_cutoff_time"], info["e1_avg_cutoff_time"], info["v_avg_cutoff_time"]
            )
        )

        observation = next_observation


if __name__ == "__main__":
    # test_env()
    loop_test()

