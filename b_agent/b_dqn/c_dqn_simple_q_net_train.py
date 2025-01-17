import torch
import os, sys
from pathlib import Path

# os.environ['WANDB_BASE_URL'] = 'http://localhost:8080'

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) 
assert BASE_PATH.endswith("link_quantum_network"), BASE_PATH
sys.path.append(BASE_PATH)

from a_env.simple_quantum_network import SimpleQuantumNetworkEnv
from b_agent.b_dqn.a_qnet import QNet
from b_agent.b_dqn.b_dqn_train_test import DqnTrainer

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "Simple_Q_Net"

    env = SimpleQuantumNetworkEnv(max_step=300)
    valid_env = SimpleQuantumNetworkEnv(max_step=300)

    print("probability_entanglement('e0'): {0}".format(env.probability_entanglement("e0")))
    print("probability_entanglement('e1'): {0}".format(env.probability_entanglement("e1")))
    print("probability_valid_state(age=0): {0}".format(env.probability_valid_state(age=0)))
    print("probability_valid_state(age=7): {0}".format(env.probability_valid_state(age=7)))
    print("probability_valid_state(age=10): {0}".format(env.probability_valid_state(age=10)))

    config = {
        "env_name": ENV_NAME,                             # 환경의 이름
        "max_num_episodes": 2000,                          # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                 # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.000001,                          # 학습율
        "gamma": 0.99,                                    # 감가율
        "steps_between_train": 2,                         # 훈련 사이의 환경 스텝 수
        "replay_buffer_size": 300_000,                    # 리플레이 버퍼 사이즈
        "epsilon_start": 0.99,                            # Epsilon 초기 값
        "epsilon_end": 0.01,                              # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.7,          # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 1,                     # Episode 통계 출력에 관한 에피소드 간격
        "target_sync_time_steps_interval": 1000,           # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "validation_time_steps_interval": 10_000,         # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,                     # 검증에 수행하는 에피소드 횟수

        "episode_reward_avg_solved": 100,                 # 훈련 종료를 위한 검증 에피소드 리워드의 Average

        "early_stop_patience": 100,
        "early_stop_delta": 0.00001
    }

    qnet = QNet(n_features=6, action_space=env.action_space)
    target_qnet = QNet(n_features=6, action_space=env.action_space)

    use_wandb = True
    dqn = DqnTrainer(
        env=env, valid_env=valid_env, qnet=qnet, target_qnet=target_qnet, config=config, use_wandb=use_wandb,
        current_dir=CURRENT_DIR
    )
    dqn.train_loop()


if __name__ == "__main__":
    main()
