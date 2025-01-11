# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import gymnasium as gym
import os

from a_env.simple_quantum_network import SimpleQuantumNetworkEnv
from a_qnet import QNet

from b_dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    ENV_NAME = "Simple_Q_Net"

    test_env = SimpleQuantumNetworkEnv(max_step=1_000)

    qnet = QNet(n_features=6, action_space=test_env.action_space)

    dqn_tester = DqnTester(
        env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR
    )
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()