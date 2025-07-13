from stable_baselines3 import PPO
from torch import nn
from pathlib import Path
import sys

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)
assert BASE_PATH.endswith("link_quantum_network"), BASE_PATH
sys.path.append(BASE_PATH)

from a_env.simple_quantum_network import SimpleQuantumNetworkEnv
from b_agent.c_stable_baseline3.a_stable_baseline_callback import EpisodeMetricsLoggingCallback


def main():
    train_env = SimpleQuantumNetworkEnv(max_step=300)
    validation_env = SimpleQuantumNetworkEnv(max_step=300)

    print("probability_entanglement('e0'): {0}".format(train_env.probability_entanglement("e0")))
    print("probability_entanglement('e1'): {0}".format(train_env.probability_entanglement("e1")))
    print("probability_valid_state(age=0): {0}".format(train_env.probability_valid_state(age=0)))
    print("probability_valid_state(age=7): {0}".format(train_env.probability_valid_state(age=7)))
    print("probability_valid_state(age=10): {0}".format(train_env.probability_valid_state(age=10)))

    # 커스텀 네트워크를 사용하여 모델 생성
    policy_kwargs = dict(
        activation_fn=nn.LeakyReLU,
        net_arch=dict(
            pi=[256, 256, 256],
            vf=[256, 256, 256]
        )
    )
    model = PPO(
        "MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=train_env,
        batch_size=1024,
        ent_coef=0.1,
        vf_coef=1.0,
        clip_range=0.3,
        learning_rate=0.00001,
        verbose=1
    )
    print(model.policy)

    use_wandb = True

    model.learn(
        total_timesteps=10_000_000,
        callback=EpisodeMetricsLoggingCallback(
            validation_env=validation_env,
            use_wandb=use_wandb,
            validation_step_frequency=20_000,
            num_validation_episodes=3
        )
    )

    model.save("ppo_simple_quantum_network")

if __name__ == "__main__":
    main()