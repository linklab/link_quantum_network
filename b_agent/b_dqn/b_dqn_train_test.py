# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import collections

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 6), (32, 6)

        actions = np.array(actions)
        # actions.shape: (32, 3)

        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        # actions.shape: (32, 3) --> (3, 32)
        return observations, actions.T, next_observations, rewards, dones


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, config, checkpoint_file_path, run_time_str):
        self.counter = 0
        self.config = config
        self.checkpoint_file_path = checkpoint_file_path
        self.run_time_str = run_time_str

        self.latest_file_path = os.path.join(
            self.checkpoint_file_path, f"{self.config['env_name']}_checkpoint_latest.pt"
        )

        self.max_validation_episode_reward_avg = None

    def check_and_save(self, new_validation_episode_reward_avg, model):
        early_stop = False

        if self.max_validation_episode_reward_avg is None:
            self.max_validation_episode_reward_avg = new_validation_episode_reward_avg
            message = f'Early stopping is stated!'
        elif new_validation_episode_reward_avg > self.max_validation_episode_reward_avg - self.config["early_stop_delta"]:
            message = f'V_loss decreased ({self.max_validation_episode_reward_avg:7.5f} --> {new_validation_episode_reward_avg:7.5f}). Saving model...'
            self.save_checkpoint(new_validation_episode_reward_avg, model)
            self.max_validation_episode_reward_avg = new_validation_episode_reward_avg
            self.counter = 0
        else:
            self.counter += 1
            message = f'Early stopping counter: {self.counter} out of {self.config["early_stop_patience"]}'
            if self.counter >= self.config["early_stop_patience"]:
                early_stop = True
                message += " *** TRAIN EARLY STOPPED! ***"

        return message, early_stop

    def save_checkpoint(self, new_validation_episode_reward_avg, model):
        file_path = os.path.join(
            self.checkpoint_file_path,
            f"{self.config['env_name']}_{new_validation_episode_reward_avg:6.2f}_checkpoint_{self.run_time_str}.pt"
        )

        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file_path)
        torch.save(model.state_dict(), self.latest_file_path)

        self.max_validation_episode_reward_avg = new_validation_episode_reward_avg


class DqnTrainer:
    def __init__(
            self, env, valid_env, qnet, target_qnet, config: dict, use_wandb: bool, current_dir
    ):
        self.env = env
        self.valid_env = valid_env
        self.use_wandb = use_wandb
        self.config = config

        print(self.config)

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if self.use_wandb:
            self.wandb = wandb.init(
                project="DQN_{0}".format(self.config["env_name"]), name=self.current_time, config=config
            )

        self.epsilon_scheduled_last_episode = self.config["max_num_episodes"] * self.config["epsilon_final_scheduled_percent"]

        # network
        self.qnet = qnet
        self.target_qnet = target_qnet
        self.target_qnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.config["learning_rate"])

        # agent
        self.replay_buffer = ReplayBuffer(self.config["replay_buffer_size"])

        self.time_steps = 0
        self.training_time_steps = 0

        self.total_train_start_time = None

    def epsilon_scheduled(self, current_episode: int) -> float:
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)
        epsilon_span = self.config["epsilon_start"] - self.config["epsilon_end"]

        epsilon = min(self.config["epsilon_start"] - fraction * epsilon_span, self.config["epsilon_start"])
        return epsilon

    def train_loop(self):
        loss = 0.0

        early_stopping = EarlyStopping(
            config=self.config, checkpoint_file_path=self.model_dir,
            run_time_str=datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
        )

        self.total_train_start_time = time.time()

        validation_episode_reward_avg = None

        is_terminated = False

        for n_episode in range(1, self.config["max_num_episodes"] + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0

            observation, info = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.qnet.get_action(observation, epsilon)

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.config["steps_between_train"] == 0 and self.time_steps > self.config["batch_size"]:
                    loss = self.train()

                if self.time_steps % self.config["validation_time_steps_interval"] == 0:
                    validation_episode_reward_avg = self.validate()
                    message, is_terminated = early_stopping.check_and_save(validation_episode_reward_avg, self.qnet)
                    print(message)
                    # if validation_episode_reward_avg > self.config["episode_reward_avg_solved"]:
                    #     print("Solved in {0:,} time steps ({1:,} training steps)!".format(
                    #         self.time_steps, self.training_time_steps
                    #     ))
                    #     self.model_save(validation_episode_reward_avg)
                    #     is_terminated = True
                    #     break

            if validation_episode_reward_avg is None:
                validation_episode_reward_avg = episode_reward

            if n_episode % self.config["print_episode_interval"] == 0:
                print(
                    "[Epi. {:3,}, T. Steps {:6,}]".format(n_episode, self.time_steps),
                    "Epi. R.: {:>3.1f},".format(episode_reward),
                    "R. Buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.4f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:5,}".format(self.training_time_steps),
                    "S.P.: {:5.3f}".format(info["avg_swap_prob"]),
                    "age: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                        info["e0_avg_age"], info["e1_avg_age"], info["v_avg_age"]
                    ),
                    "cutoff_time: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                        info["e0_avg_cutoff_time"], info["e1_avg_cutoff_time"], info["v_avg_cutoff_time"]
                    )
                )

            if self.use_wandb:
                self.wandb.log(
                    {
                        "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                            self.config["validation_num_episodes"]
                        ): validation_episode_reward_avg,
                        "[TRAIN] Episode Reward": episode_reward,
                        "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                        "[TRAIN] Epsilon": epsilon,
                        "[TRAIN] Replay buffer": self.replay_buffer.size(),
                        "[TRAIN] Swap Probability": info["avg_swap_prob"],
                        "[TRAIN] E0 Average Age": info["e0_avg_age"],
                        "[TRAIN] E1 Average Age": info["e1_avg_age"],
                        "[TRAIN] V Average Age": info["v_avg_age"],
                        "[TRAIN] E0 Cutoff Time": info["e0_avg_cutoff_time"],
                        "[TRAIN] E1 Cutoff Time": info["e1_avg_cutoff_time"],
                        "[TRAIN] V Cutoff Time": info["v_avg_cutoff_time"],
                        "Training Episode": n_episode,
                        "Training Steps": self.training_time_steps,
                    }
                )

            if is_terminated:
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def train(self) -> float:
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.config["batch_size"])

        # observations.shape: torch.Size([32, 6]),
        # actions.shape: torch.Size([3, 32]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch
        
        multi_q_out = self.qnet(observations)
        with torch.no_grad():
            multi_q_prime_out = self.target_qnet(next_observations)

        total_loss = 0

        for idx, (q_out, q_prime_out) in enumerate(zip(multi_q_out, multi_q_prime_out)):
            q_values = q_out.gather(dim=-1, index=actions[idx].unsqueeze(dim=-1))

            with torch.no_grad():
                max_q_prime = q_prime_out.max(dim=1, keepdim=True).values
                max_q_prime[dones] = 0.0

                # target_state_action_values.shape: torch.Size([32, 1])
                targets = rewards + self.config["gamma"] * max_q_prime

            # loss is just scalar torch value
            total_loss += F.mse_loss(targets.detach(), q_values)
        
        total_loss /= len(multi_q_out)

        # print("observations.shape: {0}, actions.shape: {1}, "
        #       "next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape,
        #     next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("state_action_values.shape: {0}".format(state_action_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_state_action_values.shape: {0}".format(
        #     target_state_action_values.shape
        # ))
        # print("loss.shape: {0}".format(loss.shape))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # sync
        if self.time_steps % self.config["target_sync_time_steps_interval"] == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        return total_loss.item()

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        self.qnet.eval()

        for i in range(self.config["validation_num_episodes"]):
            episode_reward = 0

            observation, _ = self.valid_env.reset()

            done = False

            while not done:
                action = self.qnet.get_action(observation, epsilon=0.0)

                next_observation, reward, terminated, truncated, _ = self.valid_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        episode_reward_avg = np.average(episode_reward_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_avg, total_training_time
            )
        )
        self.qnet.train()        
        return episode_reward_avg


class DqnTester:
    def __init__(self, env: gym.Env, qnet, env_name, current_dir):
        self.env = env

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.video_dir = os.path.join(current_dir, "videos")
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)

        self.env = gym.wrappers.RecordVideo(
            env=self.env, video_folder=self.video_dir,
            name_prefix="dqn_{0}_test_video".format(env_name)
        )

        self.qnet = qnet

        model_params = torch.load(
            os.path.join(self.model_dir, "dqn_{0}_latest.pth".format(env_name)),
            weights_only=True,
            map_location=DEVICE
        )
        self.qnet.load_state_dict(model_params)
        self.qnet.eval()

    def test(self):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = self.env.reset()
        time_steps = 0

        done = False

        while not done:
            time_steps += 1
            action = self.qnet.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        self.env.close()
        print("[TOAL_STEPS: {0:3d}, EPISODE REWARD: {1:4.1f}".format(time_steps, episode_reward))
