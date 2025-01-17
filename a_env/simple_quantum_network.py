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
            last_entanglement_try_timestep=-1
    ):
        self.node0 = node0
        self.node1 = node1
        self.distance = distance
        self.entanglement = entanglement
        self.age = age
        self.last_entanglement_try_timestep = last_entanglement_try_timestep

        self.age_list = []
        self.cutoff_try_time_list = []

        self.number_of_entanglement_state = 0

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
    optical_bsm_efficiency = 0.39
    probability_photon_detection = 0.7
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

        info = {
            "avg_swap_prob": 0.0,
            "swap_prob_list": []
        }

        for edge_idx in ["e0", "e1", "v"]:
            key = "{0}_avg_flat_age".format(edge_idx)
            info[key] = 0.0

            key = "{0}_avg_entanglement_age".format(edge_idx)
            info[key] = 0.0

            key = "{0}_age_list".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].age_list
            info[key] = value

            key = "{0}_entanglement_state_rate".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].number_of_entanglement_state / self.max_step
            info[key] = value

            key = "{0}_avg_cutoff_try_time".format(edge_idx)
            if len(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list) == 0:
                value = 0.0
            else:
                value = np.mean(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list)
            info[key] = value

            key = "{0}_cutoff_try_list".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list
            info[key] = value

        return self.current_observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.current_step += 1

        # action: [e0_action, e1_action, v_action]
        e0_action, e1_action, v_action = action

        # probability_swap_list 추가
        self.probability_swap_list.append(self.probability_swap())

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
            self.update_cutoff_try_time(edge_idx="e0")

            self.try_entanglement(edge_idx="e0")

        else:
            raise ValueError()

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
            self.update_cutoff_try_time(edge_idx="e1")

            self.try_entanglement(edge_idx="e1")

        else:
            raise ValueError()

        #######################
        # v action processing #
        #######################
        reward = 0.0

        if v_action == 0:
            if self.quantum_network.edge_dict["v"].entanglement == 1:
                self.aging_entanglement(edge_idx="v")

            if self.quantum_network.edge_dict["v"].age == self.max_entanglement_age:
                self.quantum_network.edge_dict["v"].entanglement = 0
                self.quantum_network.edge_dict["v"].age = -1

        elif v_action == 1:
            self.update_cutoff_try_time(edge_idx="v")

            reward = self.try_swap()

        else:
            raise ValueError()

        for edge_idx in ["e0", "e1", "v"]:
            self.quantum_network.edge_dict[edge_idx].age_list.append(
                self.quantum_network.edge_dict[edge_idx].age
            )

        # 현재 observation 초기화
        self.current_observation = self.get_observation()
        next_observation = self.current_observation

        info = {
            "avg_swap_prob": np.mean(self.probability_swap_list),
            "swap_prob_list": self.probability_swap_list
        }

        terminated = truncated = False

        if self.current_step == self.max_step:
            terminated = True
            self.config_info(info)

        return next_observation, reward, terminated, truncated, info

    def config_info(self, info):
        for edge_idx in ["e0", "e1", "v"]:
            age_list_replaced = [0 if x == -1 else x for x in self.quantum_network.edge_dict[edge_idx].age_list]

            max_age_list = []
            current_max = None
            for value in self.quantum_network.edge_dict[edge_idx].age_list:
                if value == -1:
                    # 현재 수열이 종료된 경우
                    if current_max is not None:
                        max_age_list.append(current_max)
                    current_max = None  # 초기화
                else:
                    # 연속 수열 내에서 최대값 업데이트
                    if current_max is None or value > current_max:
                        current_max = value

            # 마지막 수열이 끝나지 않았을 경우 처리
            if current_max is not None:
                max_age_list.append(current_max)

            key = "{0}_avg_flat_age".format(edge_idx)
            if len(age_list_replaced) == 0:
                value = 0.0
            else:
                value = np.mean(age_list_replaced)
            info[key] = value

            key = "{0}_avg_entanglement_age".format(edge_idx)
            if len(max_age_list) == 0:
                value = 0.0
            else:
                value = np.mean(max_age_list)
            info[key] = value

            key = "{0}_age_list".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].age_list
            info[key] = value

            key = "{0}_entanglement_state_rate".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].number_of_entanglement_state / self.max_step
            info[key] = value

            key = "{0}_avg_cutoff_try_time".format(edge_idx)
            if len(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list) == 0:
                value = 0.0
            else:
                value = np.mean(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list)
            info[key] = value

            key = "{0}_cutoff_try_list".format(edge_idx)
            value = self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list
            info[key] = value

    def try_entanglement(self, edge_idx):
        if random.random() <= self.probability_entanglement(edge_idx=edge_idx):
            self.quantum_network.edge_dict[edge_idx].entanglement = 1
            self.quantum_network.edge_dict[edge_idx].age = 0
            self.quantum_network.edge_dict[edge_idx].number_of_entanglement_state += 1
        else:
            self.quantum_network.edge_dict[edge_idx].entanglement = 0
            self.quantum_network.edge_dict[edge_idx].age = -1

        self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep = self.current_step

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
                self.quantum_network.edge_dict["v"].number_of_entanglement_state += 1
                reward = 1.0
            else:
                self.quantum_network.edge_dict["v"].entanglement = 0
                self.quantum_network.edge_dict["v"].age = -1

            self.quantum_network.edge_dict["e0"].entanglement = 0
            self.quantum_network.edge_dict["e0"].age = -1

            self.quantum_network.edge_dict["e1"].entanglement = 0
            self.quantum_network.edge_dict["e1"].age = -1
        else:
            if self.quantum_network.edge_dict["v"].entanglement == 1:
                self.quantum_network.edge_dict["v"].age += 1

        self.quantum_network.edge_dict["v"].last_entanglement_try_timestep = self.current_step

        return reward

    def update_cutoff_try_time(self, edge_idx):
        if self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep != -1:
            self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list.append(
                self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep
            )

    # def update_edge_age(self, edge_idx):
    #     entanglement_success_after_entanglement_conditions = [
    #         self.quantum_network.edge_dict[edge_idx].last_entanglement_success_timestep != -1,
    #         self.quantum_network.edge_dict[edge_idx].age != -1
    #     ]
    #     if all(entanglement_success_after_entanglement_conditions):
    #         self.quantum_network.edge_dict[edge_idx].age_list.append(
    #             self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_success_timestep
    #         )

    # def update_edge_age(self, edge_idx):
    #     entanglement_success_after_entanglement_conditions = [
    #         self.quantum_network.edge_dict[edge_idx].last_entanglement_success_timestep != -1,
    #         self.quantum_network.edge_dict[edge_idx].age != -1
    #     ]
    #     if all(entanglement_success_after_entanglement_conditions):
    #         self.quantum_network.edge_dict[edge_idx].age_list.append(
    #             self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_success_timestep
    #         )
    #         self.quantum_network.edge_dict[edge_idx].average_age = np.mean(
    #             self.quantum_network.edge_dict[edge_idx].age_list
    #         )

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
        self.quantum_network.edge_dict[edge_idx].number_of_entanglement_state += 1

    def probability_entanglement(self, edge_idx):
        edge_distance = self.quantum_network.edge_dict[edge_idx].distance

        prob_entanglement = (1.0 / 2.0) * self.optical_bsm_efficiency * \
                            (self.probability_photon_detection ** 2) * \
                            math.e ** (-1.0 * edge_distance / self.attenuation_length_of_the_optical_fiber)

        return prob_entanglement

    def probability_swap(self):
        max_age = max(
            self.quantum_network.edge_dict["e0"].age,
            self.quantum_network.edge_dict["e1"].age
        )
        if max_age == -1:
            prob_swap = 0.0
        else:
            prob_swap = self.probability_valid_state(max_age)

        return prob_swap

    # def probability_swap(self):
    #     entanglement_conditions = [
    #         self.quantum_network.edge_dict["e0"].entanglement == 1,
    #         self.quantum_network.edge_dict["e1"].entanglement == 1
    #     ]
    #     if all(entanglement_conditions):
    #         max_age = max(
    #             self.quantum_network.edge_dict["e0"].age,
    #             self.quantum_network.edge_dict["e1"].age
    #         )
    #         prob_swap = self.probability_valid_state(max_age)
    #     else:
    #         prob_swap = 0.0
    #
    #     return prob_swap

    def probability_valid_state(self, age):
        prob_valid_entanglement = self.probability_init_memory * math.e ** (-1.0 * age / self.t_0)

        return prob_valid_entanglement

    def render(self):
        pass


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
    print("probability_valid_state(-1): {0}".format(env.probability_valid_state(age=-1)))
    print("probability_valid_state(0): {0}".format(env.probability_valid_state(age=0)))
    print("probability_valid_state(7): {0}".format(env.probability_valid_state(age=7)))
    print("probability_valid_state(10): {0}".format(env.probability_valid_state(age=10)))
    print(env.probability_swap())
    print(env.try_entanglement("e0"))
    print()


def loop_test():
    env = SimpleQuantumNetworkEnv(max_step=300)
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
            "cutoff_try_time: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                info["e0_avg_cutoff_try_time"], info["e1_avg_cutoff_try_time"], info["v_avg_cutoff_try_time"]
            )
        )
        print("e0_age_list: ", info["e0_age_list"])
        print("e0_cutoff_try_list: ", info["e0_cutoff_try_list"])
        print("e1_age_list: ", info["e1_age_list"])
        print("e1_cutoff_try_list: ", info["e1_cutoff_try_list"])
        print("v_age_list: ", info["v_age_list"])
        print("v_cutoff_try_list: ", info["v_cutoff_try_list"])
        print()

        observation = next_observation


if __name__ == "__main__":
    test_env()
    #loop_test()

