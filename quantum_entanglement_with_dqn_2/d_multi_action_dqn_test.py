import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from quantum_entanglement_with_dqn_2.a_quantum_network_environment import QuantumNetworkEnv
from b_multi_action_qnet import QNet, MODEL_DIR


def test(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_play(num_episodes, env_name):
    env = QuantumNetworkEnv()

    q = QNet(n_features=6, n_multi_actions=[2, 2, 2])
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_{0}_latest.pth".format(env_name)))
    q.load_state_dict(model_params)

    test(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "QuantumNetwork"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
