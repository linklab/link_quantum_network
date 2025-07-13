# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time
from datetime import datetime

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

        self.max_episode_reward_lst_avg = None

    def check_and_save(self, new_episode_reward_lst_avg, model):
        early_stop = False

        if self.max_episode_reward_lst_avg is None:
            self.max_episode_reward_lst_avg = new_episode_reward_lst_avg
            message = f'Early stopping is stated!'
        elif new_episode_reward_lst_avg > self.max_episode_reward_lst_avg + self.config["early_stop_delta"]:
            message = f'V_loss decreased ({self.max_episode_reward_lst_avg:7.5f} --> {new_episode_reward_lst_avg:7.5f}). Saving model...'
            self.save_checkpoint(new_episode_reward_lst_avg, model)
            self.max_episode_reward_lst_avg = new_episode_reward_lst_avg
            self.counter = 0
        else:
            self.counter += 1
            message = f'Early stopping counter: {self.counter} out of {self.config["early_stop_patience"]}'
            if self.counter >= self.config["early_stop_patience"]:
                early_stop = True
                message += " *** TRAIN EARLY STOPPED! ***"

        return message, early_stop

    def save_checkpoint(self, new_episode_reward_lst_avg, model):
        file_path = os.path.join(
            self.checkpoint_file_path,
            f"{self.config['env_name']}_{self.run_time_str}_checkpoint_{new_episode_reward_lst_avg:5.2f}.pt"
        )

        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), file_path)
        torch.save(model.state_dict(), self.latest_file_path)

        self.max_episode_reward_lst_avg = new_episode_reward_lst_avg


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

        episode_reward_lst_avg = swap_prob_lst_avg = \
            e0_entanglement_age_lst_avg = \
            e1_entanglement_age_lst_avg = \
            v_entanglement_age_lst_avg = \
            e0_entanglement_state_fraction_lst_avg = e1_entanglement_state_fraction_lst_avg = v_entanglement_state_fraction_lst_avg = \
            e0_cutoff_try_time_lst_avg = e1_cutoff_try_time_lst_avg = v_cutoff_try_time_lst_avg = None

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
                    (episode_reward_lst_avg,
                    swap_prob_lst_avg,

                    e0_entanglement_age_lst_avg,
                    e1_entanglement_age_lst_avg,
                    v_entanglement_age_lst_avg,

                    e0_entanglement_state_fraction_lst_avg,
                    e1_entanglement_state_fraction_lst_avg,
                    v_entanglement_state_fraction_lst_avg,

                    e0_cutoff_try_time_lst_avg,
                    e1_cutoff_try_time_lst_avg,
                    v_cutoff_try_time_lst_avg) = self.validate()

                    message, is_terminated = early_stopping.check_and_save(episode_reward_lst_avg, self.qnet)
                    print(message)

                    # if episode_reward_lst_avg > self.config["episode_reward_solved"]:
                    #     print("Solved in {0:,} time steps ({1:,} training steps)!".format(
                    #         self.time_steps, self.training_time_steps
                    #     ))
                    #     self.model_save(episode_reward_lst_avg)
                    #     is_terminated = True
                    #     break

            if episode_reward_lst_avg is None:
                episode_reward_lst_avg = episode_reward

            if n_episode % self.config["print_episode_interval"] == 0:
                print(
                    "[Epi. {:3,}, T. Steps {:6,}]".format(n_episode, self.time_steps),
                    "Epi. R.: {:>3.1f},".format(episode_reward),
                    "R. Buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.4f},".format(loss),
                    "Eps.: {:4.2f},".format(epsilon),
                    "Train Steps: {:5,}".format(self.training_time_steps),
                    "S.P.: {:5.3f}".format(info["swap_prob"]),
                    "entanglement_age: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                        info["e0_entanglement_age"], info["e1_entanglement_age"], info["v_entanglement_age"]
                    ),
                    "cutoff_try_time: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                        info["e0_cutoff_try_time"], info["e1_cutoff_try_time"], info["v_cutoff_try_time"]
                    )
                )

            if n_episode % (self.config["print_episode_interval"] * 10) == 0:
                print("swap_prob_list: ", info["swap_prob_list"])
                print("e0_age_list: ", info["e0_age_list"])
                print("e0_cutoff_try_list: ", info["e0_cutoff_try_list"])
                print("e1_age_list: ", info["e1_age_list"])
                print("e1_cutoff_try_list: ", info["e1_cutoff_try_list"])
                print("v_age_list: ", info["v_age_list"])
                print("v_cutoff_try_list: ", info["v_cutoff_try_list"])

            if self.use_wandb:
                self.wandb.log(
                    {
                        "VALIDATION/Episode Reward ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            episode_reward_lst_avg,

                        "VALIDATION/Swap Prob. ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            swap_prob_lst_avg,

                        "VALIDATION/E0 Entang. Age ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e0_entanglement_age_lst_avg,
                        "VALIDATION/E1 Entang. Age({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e1_entanglement_age_lst_avg,
                        "VALIDATION/V Entang. Age({0} Episodes)".format(self.config["validation_num_episodes"]):
                            v_entanglement_age_lst_avg,

                        "VALIDATION/E0 Entang. State Fraction ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e0_entanglement_state_fraction_lst_avg,
                        "VALIDATION/E1 Entang. State Fraction ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e1_entanglement_state_fraction_lst_avg,
                        "VALIDATION/V Entang. State Fraction ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            v_entanglement_state_fraction_lst_avg,

                        "VALIDATION/E0 Cutoff Try Time ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e0_cutoff_try_time_lst_avg,
                        "VALIDATION/E1 Cutoff Try Time ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            e1_cutoff_try_time_lst_avg,
                        "VALIDATION/V Cutoff Try Time ({0} Episodes)".format(self.config["validation_num_episodes"]):
                            v_cutoff_try_time_lst_avg,

                        "TRAIN/Episode Reward": episode_reward,
                        "TRAIN/Loss": loss if loss != 0.0 else 0.0,
                        "TRAIN/Epsilon": epsilon,
                        "TRAIN/Replay buffer": self.replay_buffer.size(),

                        "TRAIN/Swap Prob.": info["swap_prob"],

                        "TRAIN/E0 Entang. Age": info["e0_entanglement_age"],
                        "TRAIN/E1 Entang. Age": info["e1_entanglement_age"],
                        "TRAIN/V Entang. Age": info["v_entanglement_age"],

                        "TRAIN/E0 Entang. State Fraction": info["e0_entanglement_state_fraction"],
                        "TRAIN/E1 Entang. State Fraction": info["e1_entanglement_state_fraction"],
                        "TRAIN/V Entang. State Fraction": info["v_entanglement_state_fraction"],

                        "TRAIN/E0 Cutoff Try Time": info["e0_cutoff_try_time"],
                        "TRAIN/E1 Cutoff Try Time": info["e1_cutoff_try_time"],
                        "TRAIN/V Cutoff Try Time": info["v_cutoff_try_time"],

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
        multi_next_q_out = self.qnet(next_observations)
        with torch.no_grad():
            multi_q_prime_out = self.target_qnet(next_observations)

        total_loss = 0

        for idx, (q_out, next_q_out, q_prime_out) in enumerate(zip(multi_q_out, multi_next_q_out, multi_q_prime_out)):
            q_values = q_out.gather(dim=-1, index=actions[idx].unsqueeze(dim=-1))

            ## Double DQN
            with torch.no_grad():
                target_argmax_action = torch.argmax(next_q_out, dim=-1, keepdim=True)
                max_q_prime = q_prime_out.gather(dim=-1, index=target_argmax_action)
                max_q_prime[dones] = 0.0

                # target_state_action_values.shape: torch.Size([32, 1])
                targets = rewards + self.config["gamma"] * max_q_prime

                # print(q_out.shape, q_prime_out.shape, next_q_out.shape, target_argmax_action.shape, targets.shape, rewards.shape, max_q_prime.shape, "!!!!!!!!!!")

            # loss is just scalar torch value
            total_loss += F.huber_loss(targets.detach(), q_values)
        
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

        max_norm = 1.0  # Gradient norm의 최대값
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm)

        self.optimizer.step()

        # sync
        if self.time_steps % self.config["target_sync_time_steps_interval"] == 0:
            self.target_qnet.load_state_dict(self.qnet.state_dict())

        return total_loss.item()

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        swap_prob_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)

        e0_entanglement_age_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        e1_entanglement_age_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        v_entanglement_age_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)

        e0_entanglement_state_fraction_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        e1_entanglement_state_fraction_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        v_entanglement_state_fraction_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)

        e0_cutoff_try_time_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        e1_cutoff_try_time_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)
        v_cutoff_try_time_lst = np.zeros(shape=(self.config["validation_num_episodes"],), dtype=float)

        self.qnet.eval()

        for i in range(self.config["validation_num_episodes"]):
            episode_reward = 0

            observation, info = self.valid_env.reset()

            done = False

            while not done:
                action = self.qnet.get_action(observation, epsilon=0.0)
                next_observation, reward, terminated, truncated, info = self.valid_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward
            swap_prob_lst[i] = info["swap_prob"]

            e0_entanglement_age_lst[i] = info["e0_entanglement_age"]
            e1_entanglement_age_lst[i] = info["e1_entanglement_age"]
            v_entanglement_age_lst[i] = info["v_entanglement_age"]

            e0_entanglement_state_fraction_lst[i] = info["e0_entanglement_state_fraction"]
            e1_entanglement_state_fraction_lst[i] = info["e1_entanglement_state_fraction"]
            v_entanglement_state_fraction_lst[i] = info["v_entanglement_state_fraction"]

            e0_cutoff_try_time_lst[i] = info["e0_cutoff_try_time"]
            e1_cutoff_try_time_lst[i] = info["e1_cutoff_try_time"]
            v_cutoff_try_time_lst[i] = info["v_cutoff_try_time"]

        episode_reward_lst_avg = np.average(episode_reward_lst)
        swap_prob_lst_avg = np.average(swap_prob_lst)

        e0_entanglement_age_lst_avg = np.average(e0_entanglement_age_lst)
        e1_entanglement_age_lst_avg = np.average(e1_entanglement_age_lst)
        v_entanglement_age_lst_avg = np.average(v_entanglement_age_lst)

        e0_entanglement_state_fraction_lst_avg = np.average(e0_entanglement_state_fraction_lst)
        e1_entanglement_state_fraction_lst_avg = np.average(e1_entanglement_state_fraction_lst)
        v_entanglement_state_fraction_lst_avg = np.average(v_entanglement_state_fraction_lst)

        e0_cutoff_try_time_lst_avg = np.average(e0_cutoff_try_time_lst)
        e1_cutoff_try_time_lst_avg = np.average(e1_cutoff_try_time_lst)
        v_cutoff_try_time_lst_avg = np.average(v_cutoff_try_time_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_lst_avg, total_training_time
            )
        )
        self.qnet.train()        
        return (episode_reward_lst_avg,
                swap_prob_lst_avg,

                e0_entanglement_age_lst_avg,
                e1_entanglement_age_lst_avg,
                v_entanglement_age_lst_avg,

                e0_entanglement_state_fraction_lst_avg,
                e1_entanglement_state_fraction_lst_avg,
                v_entanglement_state_fraction_lst_avg,

                e0_cutoff_try_time_lst_avg,
                e1_cutoff_try_time_lst_avg,
                v_cutoff_try_time_lst_avg)


class DqnTester:
    def __init__(self, env: gym.Env, qnet, env_name, current_dir):
        self.env = env

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.video_dir = os.path.join(current_dir, "videos")
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)

        self.qnet = qnet

        model_params = torch.load(
            os.path.join(self.model_dir, "{0}_checkpoint_latest.pt".format(env_name)),
            weights_only=True,
            map_location=DEVICE
        )
        self.qnet.load_state_dict(model_params)
        self.qnet.eval()

    def test(self):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, info = self.env.reset()
        time_steps = 0

        done = False

        while not done:
            time_steps += 1
            action = self.qnet.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, info = self.env.step(action)

            print(
                "[Step: {0:>3}] Ob.: {1:<22} Act: {2} N.Ob.: {3:<22} "
                "R.: {4} Term.: {5:<5}".format(
                    time_steps,
                    str(observation),
                    action,
                    str(next_observation),
                    reward,
                    str(terminated),
                ),
                end=" "
            )
            print(
                "S.P.: {:5.3f}".format(info["swap_prob"]),
                "age: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                    info["e0_age"], info["e1_age"], info["v_age"]
                ),
                "cutoff_try_time: {0:3.1f}/{1:3.1f}/{2:3.1f}".format(
                    info["e0_cutoff_try_time"], info["e1_cutoff_try_time"], info["v_cutoff_try_time"]
                )
            )

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        self.env.close()
        print("TOTAL STEPS: {0:3d}".format(time_steps))
        print("EPISODE REWARD: {0:5.2f}".format(episode_reward))
        print("Average Swap Probability: {0:5.2f}".format(info["swap_prob"]))
        print("E0 Average Age: {0:5.2f}".format(info["e0_age"]))
        print("E1 Average Age: {0:5.2f}".format(info["e1_age"]))
        print("V Average Age: {0:5.2f}".format(info["v_age"]))
        print("E0 Average Cutoff Time: {0:5.2f}".format(info["e0_cutoff_try_time"]))
        print("E1 Average Cutoff Time: {0:5.2f}".format(info["e1_cutoff_try_time"]))
        print("V Average Cutoff Time: {0:5.2f}".format(info["v_cutoff_try_time"]))