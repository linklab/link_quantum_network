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
            self, node0, node1, distance, entanglement=0, age=-1, last_entanglement_try_timestep=-1
    ):
        self.node0 = node0
        self.node1 = node1
        self.distance = distance
        self.entanglement = entanglement
        self.age = age
        self.last_entanglement_try_timestep = last_entanglement_try_timestep

        self.age_list = []
        self.state_list = []    # not-entangled: 0, entangled: 1
        self.cutoff_try_time_list = []

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

    max_entanglement_age = 10

    running_average_window = 50

    def __init__(self, max_step):
        # s_t = [x_0, m_0, x_1, m_1, x_2, m_2]
        # x_k: {0, 1}, m_k: {-1, 0, ..., max_age}
        # x_k: 0 또는 1의 값을 가지는 이진 상태.
        # m_k: -1 (inactive) 또는 정수 값 (entanglement age).
        self.observation_space = gym.spaces.Box(
            low=np.asarray([0.0, -1.0, 0.0, -1.0, 0.0, -1.0]),
            high=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float64
        )

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
            self.quantum_network.edge_dict["e0"].entanglement,
            self.quantum_network.edge_dict["e0"].age / self.max_entanglement_age,
            self.quantum_network.edge_dict["e1"].entanglement,
            self.quantum_network.edge_dict["e1"].age / self.max_entanglement_age,
            self.quantum_network.edge_dict["v"].entanglement,
            self.quantum_network.edge_dict["v"].age / self.max_entanglement_age
        ]

        return np.asarray(observation)

    def reset(self, seed=0):
        self.quantum_network = SimpleQuantumNetwork()

        # 현재 observation 초기화
        self.current_observation = self.get_observation()
        self.current_step = 0
        self.probability_swap_list.clear()

        info = self.config_info()

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

        # 단순 wait
        if e0_action == 0:
            self.do_wait(edge_idx="e0")
        # entanglement 시도
        elif e0_action == 1:
            self.update_cutoff_try_time(edge_idx="e0")
            self.do_try_entanglement(edge_idx="e0")
        else:
            raise ValueError()

        ########################
        # e1 action processing #
        ########################
        # 단순 wait
        if e1_action == 0:
            self.do_wait(edge_idx="e1")
        # entanglement 시도
        elif e1_action == 1:
            self.update_cutoff_try_time(edge_idx="e1")
            self.do_try_entanglement(edge_idx="e1")
        else:
            raise ValueError()

        #######################
        # v action processing #
        #######################
        # 단순 wait
        if v_action == 0:
            self.do_wait(edge_idx="v")
        # entanglement 시도
        elif v_action == 1:
            self.update_cutoff_try_time(edge_idx="v")

            entanglement_conditions = [
                self.quantum_network.edge_dict["e0"].entanglement == 1,
                self.quantum_network.edge_dict["e1"].entanglement == 1
            ]
            if all(entanglement_conditions):
                self.do_try_swap()
            else:
                self.do_wait(edge_idx="v")
        else:
            raise ValueError()

        for edge_idx in ["e0", "e1", "v"]:
            self.quantum_network.edge_dict[edge_idx].state_list.append(
                self.quantum_network.edge_dict[edge_idx].entanglement
            )
            self.quantum_network.edge_dict[edge_idx].age_list.append(
                self.quantum_network.edge_dict[edge_idx].age
            )

        if self.quantum_network.edge_dict["v"].entanglement == 1:
            reward = 1.0 / self.max_step
        else:
            reward = 0.0

        # 현재 observation 초기화
        self.current_observation = self.get_observation()
        next_observation = self.current_observation

        terminated = truncated = False

        if self.current_step == self.max_step:
            terminated = True

        info = self.config_info()

        return next_observation, reward, terminated, truncated, info

    def get_average_valid_sequence(self, age_list):
        result = []
        i = 0
        n = len(age_list)

        while i < n:
            # -1은 무시
            if age_list[i] == 0:
                current_group = [0]
                j = i + 1
                expected = 1

                # 연속된 순서대로 증가하는 값들만 포함 (0,1,2,3,...)
                while j < n and age_list[j] == expected:
                    current_group.append(age_list[j])
                    expected += 1
                    j += 1

                result.append(max(current_group))
                i = j  # 다음 그룹으로 건너뜀
            else:
                i += 1
        # 평균 계산
        if result:
            avg_max = sum(result) / len(result)
        else:
            avg_max = -1  # 그룹이 없을 경우

        return avg_max

    def config_info(self):
        info = {
            "swap_prob": np.mean(self.probability_swap_list),
            "swap_prob_list": self.probability_swap_list
        }
        for edge_idx in ["e0", "e1", "v"]:
            assert len(self.quantum_network.edge_dict[edge_idx].state_list) == len(self.quantum_network.edge_dict[edge_idx].age_list), (
                edge_idx, len(self.quantum_network.edge_dict[edge_idx].state_list), len(self.quantum_network.edge_dict[edge_idx].age_list)
            )
            key = "{0}_state_list".format(edge_idx)
            info[key] = self.quantum_network.edge_dict[edge_idx].state_list

            key = "{0}_entanglement_state_fraction".format(edge_idx)
            if len(self.quantum_network.edge_dict[edge_idx].state_list) == 0:
                info[key] = 0.0
            else:
                info[key] = sum(self.quantum_network.edge_dict[edge_idx].state_list) / len(self.quantum_network.edge_dict[edge_idx].state_list)

            key = "{0}_entanglement_age".format(edge_idx)
            if len(self.quantum_network.edge_dict[edge_idx].age_list) == 0:
                info[key] = 0.0
            else:
                info[key] = self.get_average_valid_sequence(self.quantum_network.edge_dict[edge_idx].age_list)

            key = "{0}_age_list".format(edge_idx)
            info[key] = self.quantum_network.edge_dict[edge_idx].age_list

            key = "{0}_cutoff_try_time".format(edge_idx)
            if len(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list) == 0:
                info[key] = 0.0
            else:
                info[key] = np.mean(self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list)

            key = "{0}_cutoff_try_list".format(edge_idx)
            info[key] = self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list

        return info

    def do_wait(self, edge_idx):
        # age가 max_entanglement_age와 동일하면 entanglement 삭제
        if self.quantum_network.edge_dict[edge_idx].age >= self.max_entanglement_age:
            self.quantum_network.edge_dict[edge_idx].entanglement = 0
            self.quantum_network.edge_dict[edge_idx].age = -1
        # age가 유효하면
        else:
            if self.quantum_network.edge_dict[edge_idx].entanglement == 0:
                self.quantum_network.edge_dict[edge_idx].age = -1
            elif self.quantum_network.edge_dict[edge_idx].entanglement == 1:
                self.quantum_network.edge_dict[edge_idx].age += 1
            else:
                raise ValueError()

    def do_try_entanglement(self, edge_idx):
        if random.random() <= self.probability_entanglement(edge_idx=edge_idx):
            self.quantum_network.edge_dict[edge_idx].entanglement = 1
            self.quantum_network.edge_dict[edge_idx].age = 0
        else:
            self.quantum_network.edge_dict[edge_idx].entanglement = 0
            self.quantum_network.edge_dict[edge_idx].age = -1

    def do_try_swap(self):
        self.quantum_network.edge_dict["e0"].entanglement = 0
        self.quantum_network.edge_dict["e0"].age = -1

        self.quantum_network.edge_dict["e1"].entanglement = 0
        self.quantum_network.edge_dict["e1"].age = -1

        if random.random() <= self.probability_swap():
            self.quantum_network.edge_dict["v"].entanglement = 1
            self.quantum_network.edge_dict["v"].age = 0
        else:
            self.quantum_network.edge_dict["v"].entanglement = 0
            self.quantum_network.edge_dict["v"].age = -1

    def update_cutoff_try_time(self, edge_idx):
        if self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep != -1:
            self.quantum_network.edge_dict[edge_idx].cutoff_try_time_list.append(
                self.current_step - self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep
            )
        self.quantum_network.edge_dict[edge_idx].last_entanglement_try_timestep = self.current_step

    def probability_entanglement(self, edge_idx):
        # edge_distance = self.quantum_network.edge_dict[edge_idx].distance
        #
        # prob_entanglement = (1.0 / 2.0) * self.optical_bsm_efficiency * \
        #                     (self.probability_photon_detection ** 2) * \
        #                     math.e ** (-1.0 * edge_distance / self.attenuation_length_of_the_optical_fiber)

        prob_entanglement = 0.5

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
        prob_valid_entanglement = self.probability_init_memory * math.e ** (-1.0 * age / self.max_entanglement_age)

        return prob_valid_entanglement

    def render(self):
        pass

    # 2025.06.06
    def fidelity_decay_swap(self, F0, t1, t2, tau):
        F1 = F0 * np.exp(-t1 / tau)
        F2 = F0 * np.exp(-t2 / tau)
        term = ((4 * F1 - 1) / 3) * ((4 * F2 - 1) / 3)
        return 0.75 * (1 / 3 + term)


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
    env = SimpleQuantumNetworkEnv(max_step=200)
    observation, info = env.reset()

    print(observation)
    info_str = ""
    for key in info:
        info_str += key + ": {0}\n".format(info[key])
    print(info_str)

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
            "[Step: {0:>3}] Ob.: {1:<22}, Act: {2}, N.Ob.: {3:<22}, R.: {4}, Term.: {5:<5}".format(
                episode_step,
                str(observation),
                action,
                str(next_observation),
                reward,
                str(terminated),
            )
        )

        info_str = ""
        for key in info:
            info_str += key + ": {0}\n".format(info[key])
        print(info_str)

        observation = next_observation


if __name__ == "__main__":
    # test_env()
    loop_test()

