from stable_baselines3 import PPO
from a_env.simple_quantum_network import SimpleQuantumNetworkEnv
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import wandb
from datetime import datetime


class EpisodeMetricsLoggingCallback(BaseCallback):
    def __init__(self, validation_env, use_wandb, validation_step_frequency=12_000, num_validation_episodes=5, verbose=0):
        super(EpisodeMetricsLoggingCallback, self).__init__(verbose)

        self.validation_env = validation_env
        self.use_wandb = use_wandb
        self.validation_step_frequency=validation_step_frequency
        self.num_validation_episodes = num_validation_episodes

        self.episodes = 0

        self.train_episode_reward = None
        self.train_swap_prob = None
        self.train_flat_age = {}
        self.train_entanglement_age = {}
        self.train_entanglement_state_fraction = {}
        self.train_cutoff_try_time = {}

        self.validation_episode_reward = None
        self.validation_swap_prob = None
        self.validation_flat_age = {}
        self.validation_entanglement_age = {}
        self.validation_entanglement_state_fraction = {}
        self.validation_cutoff_try_time = {}

        # wandb 초기화
        if self.use_wandb:
            wandb.init(
                project="ppo_simple_quantum_network",  # 프로젝트 이름
                name=datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S"),
                mode="online" if self.use_wandb else "disabled"
            )

    def _on_step(self) -> bool:
        # 환경에서 dones 정보 확인
        if self.n_calls % self.validation_step_frequency == 0:
            self.validate()

        dones = self.locals["dones"]
        assert len(dones) == 1
        done = dones[0]

        # 에피소드가 끝날 때마다 보상과 길이 기록
        if done and self.use_wandb:
            self.episodes += 1

            learning_rate = self.locals.get("learning_rate", None)
            info = self.locals["infos"][0]

            self.train_episode_reward = info["episode"]["r"]
            self.train_swap_prob = info["swap_prob"]

            logs = self.model.logger.name_to_value

            self.send_wandb_log(info, learning_rate, logs)

        return True

    def validate(self):
        validation_episode_reward_lst = np.zeros(shape=(self.num_validation_episodes,), dtype=float)
        validation_swap_prob_lst = np.zeros(shape=(self.num_validation_episodes,), dtype=float)

        validation_flat_age_lst = {}
        validation_entanglement_age_lst = {}
        validation_entanglement_state_fraction_lst = {}
        validation_cutoff_try_time_lst = {}

        for edge_idx in ["e0", "e1", "v"]:
            validation_flat_age_lst[edge_idx] = np.zeros(shape=(self.num_validation_episodes,), dtype=float)
            validation_entanglement_age_lst[edge_idx] = np.zeros(shape=(self.num_validation_episodes,), dtype=float)
            validation_entanglement_state_fraction_lst[edge_idx] = np.zeros(shape=(self.num_validation_episodes,), dtype=float)
            validation_cutoff_try_time_lst[edge_idx] = np.zeros(shape=(self.num_validation_episodes,), dtype=float)

        for i in range(self.num_validation_episodes):
            episode_reward = 0

            observation, info = self.validation_env.reset()

            done = False

            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, info = self.validation_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            validation_episode_reward_lst[i] = episode_reward
            validation_swap_prob_lst[i] = info["swap_prob"]

            for edge_idx in ["e0", "e1", "v"]:
                validation_flat_age_lst[edge_idx][i] = info["{0}_flat_age".format(edge_idx)]
                validation_entanglement_age_lst[edge_idx][i] = info["{0}_entanglement_age".format(edge_idx)]
                validation_entanglement_state_fraction_lst[edge_idx][i] = info["{0}_entanglement_state_fraction".format(edge_idx)]
                validation_cutoff_try_time_lst[edge_idx][i] = info["{0}_cutoff_try_time".format(edge_idx)]

        self.validation_episode_reward = np.average(validation_episode_reward_lst)
        self.validation_swap_prob = np.average(validation_swap_prob_lst)

        self.validation_flat_age = {}
        self.validation_entanglement_age = {}
        self.validation_entanglement_state_fraction = {}
        self.validation_cutoff_try_time = {}

        for edge_idx in ["e0", "e1", "v"]:
            self.validation_flat_age[edge_idx] = np.average(validation_flat_age_lst[edge_idx])
            self.validation_entanglement_age[edge_idx] = np.average(validation_entanglement_age_lst[edge_idx])
            self.validation_entanglement_state_fraction[edge_idx] = np.average(validation_entanglement_state_fraction_lst[edge_idx])
            self.validation_cutoff_try_time[edge_idx] = np.average(validation_cutoff_try_time_lst[edge_idx])

        print("[Validation Episode Reward: {0}] Average: {1:.3f}, Swap Prob.: {2:.3f}".format(
            self.num_validation_episodes, self.validation_episode_reward, self.validation_swap_prob
        ))

    def send_wandb_log(self,info, learning_rate, logs):
        log_dict = {
            "episodes": self.episodes,

            "T. METRICS/Episode Reward": self.train_episode_reward,
            "T. METRICS/Swap Prob.": self.train_swap_prob,

            "V. METRICS/Episode Reward": self.validation_episode_reward if self.validation_episode_reward is not None else self.train_episode_reward,
            "V. METRICS/Swap Prob.": self.validation_swap_prob if self.validation_swap_prob is not None else self.train_swap_prob,

            "TRAIN/learning_rate": learning_rate,

            # approx_kl:
            # 정책 업데이트 전후의 정책 분포 간 근사 KL 발산(Kullback-Leibler divergence) 값
            "TRAIN/approx_kl": logs.get("train/approx_kl", None),

            # clip_fraction:
            # 정책 확률 비율(ratio = π(a|s) / π_old(a|s))이 clip_range 바깥으로 벗어난 샘플의 비율.
            # 이 값이 너무 높으면 clipping이 자주 발생해 학습이 느려짐
            "TRAIN/clip_fraction": logs.get("train/clip_fraction", None),

            # clip_range
            # 클리핑 임계값으로, ratio가 1 ± clip_range 범위를 벗어나지 못하도록 제한.
            # 일반적으로 0.1 ~ 0.3 사이의 값을 사용
            "TRAIN/clip_range": logs.get("train/clip_range", None),

            # entropy_loss
            # 	•	정책의 엔트로피 손실(정책의 랜덤성)을 나타냅니다.
            # 	•	엔트로피가 높으면 에이전트가 탐험(exploration)을 많이 하고 있음을 의미합니다.
            # 	•	PPO는 엔트로피 손실을 보상으로 추가하여 학습 초기 탐험을 유도합니다.
            # 	•	값이 음수로 나타나는 것은 손실로 정의되었기 때문입니다.
            "TRAIN/entropy_loss": logs.get("train/entropy_loss", None),

            # explained_variance
            # 	•	상태-가치 함수 예측의 성능을 나타내는 지표로, 예측된 값과 실제 보상 간의 상관관계입니다.
            # 	•	값 범위: -1에서 1.
            # 	    - 1에 가까울수록 예측이 정확함.
            # 	    - 0에 가까우면 예측과 무관함.
            # 	    - 음수는 예측이 매우 부정확함을 의미.
            # 	•	학습이 진행됨에 따라 이 값이 증가해야 합니다.
            "TRAIN/explained_variance": logs.get("train/explained_variance", None),

            # policy_gradient_loss
            # 	•	정책 네트워크의 손실로, 행동-보상 관계를 강화하기 위해 그래디언트를 통해 학습.
            # 	•	음수 값은 강화하려는 행동에서 손실로 정의되었기 때문.
            # 	•	값이 0에 가까우면 정책 업데이트의 크기가 작아짐을 의미.
            "TRAIN/policy_gradient_loss": logs.get("train/policy_gradient_loss", None),

            # value_loss
            # 	•	상태-가치 함수의 손실로, 가치 네트워크가 환경에서 반환된 실제 보상과 예상 보상 간의 차이를 최소화하려는 손실 값
            # 	•	학습이 진행됨에 따라 감소해야 합니다.
            "TRAIN/value_loss": logs.get("train/value_loss", None),

            # loss
            # 	•	총 손실 값으로, PPO의 정책 손실(Policy Loss), 값 손실(Value Loss), 엔트로피 보너스(Entropy Bonus)를 모두 포함한 손실의 합
            # 	•	값이 작아질수록 학습이 잘 진행되고 있음을 나타냅니다.
            "TRAIN/loss": logs.get("train/loss", None),
        }

        for edge_idx in ["e0", "e1", "v"]:
            log_dict["T. METRICS/{0} Flat Age".format(edge_idx)] = info["{0}_flat_age".format(edge_idx)]
            log_dict["T. METRICS/{0} Entang. Age".format(edge_idx)] = info["{0}_entanglement_age".format(edge_idx)]
            log_dict["T. METRICS/{0} Entang. State Fraction".format(edge_idx)] = info["{0}_entanglement_state_fraction".format(edge_idx)]
            log_dict["T. METRICS/{0} Cutoff Try Time".format(edge_idx)] = info["{0}_cutoff_try_time".format(edge_idx)]

            log_dict["V. METRICS/{0} Flat Age".format(edge_idx)] = self.validation_flat_age[edge_idx] if edge_idx in self.validation_flat_age else info["{0}_flat_age".format(edge_idx)]
            log_dict["V. METRICS/{0} Entang. Age".format(edge_idx)] = self.validation_entanglement_age[edge_idx] if edge_idx in self.validation_entanglement_age else info["{0}_entanglement_age".format(edge_idx)]
            log_dict["V. METRICS/{0} Entang. State Fraction".format(edge_idx)] = self.validation_entanglement_state_fraction[edge_idx] if edge_idx in self.validation_entanglement_state_fraction else info["{0}_entanglement_state_fraction".format(edge_idx)]
            log_dict["V. METRICS/{0} Cutoff Try Time".format(edge_idx)] = self.validation_cutoff_try_time[edge_idx] if edge_idx in self.validation_cutoff_try_time else info["{0}_cutoff_try_time".format(edge_idx)]

        wandb.log(log_dict)


def main():
    train_env = SimpleQuantumNetworkEnv(max_step=300)
    validation_env = SimpleQuantumNetworkEnv(max_step=300)

    print("probability_entanglement('e0'): {0}".format(train_env.probability_entanglement("e0")))
    print("probability_entanglement('e1'): {0}".format(train_env.probability_entanglement("e1")))
    print("probability_valid_state(age=0): {0}".format(train_env.probability_valid_state(age=0)))
    print("probability_valid_state(age=7): {0}".format(train_env.probability_valid_state(age=7)))
    print("probability_valid_state(age=10): {0}".format(train_env.probability_valid_state(age=10)))

    model = PPO("MlpPolicy", train_env, verbose=1)

    use_wandb = True

    model.learn(
        total_timesteps=1_000_000,
        callback=EpisodeMetricsLoggingCallback(
            validation_env=validation_env,
            use_wandb=use_wandb,
            validation_step_frequency=12_000,
            num_validation_episodes=5
        )
    )

    model.save("ppo_simple_quantum_network")

if __name__ == "__main__":
    main()