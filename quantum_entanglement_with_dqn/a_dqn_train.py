from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Create the environment
from quantum_entanglement_with_dqn.a_env import QuantumEntanglementEnv


class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        self.current_rewards += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        return True


# Create the environment
env = QuantumEntanglementEnv()

# Check the environment
check_env(env)

# Setup reward logger
reward_logger = RewardLogger()

# Train the model with reward logger
model = DQN('MlpPolicy', env, verbose=1)
# total_timesteps = 1_000_000 # 1000 episodes with 1000 timesteps each
total_timesteps = 10_000 # 10 episodes with 1000 timesteps each
model.learn(total_timesteps=total_timesteps, callback=reward_logger)

# Save the model
model.save("dqn_quantum_entanglement")

# Calculate cumulative rewards for training
training_rewards = reward_logger.episode_rewards

# Plot training rewards
plt.figure(figsize=(10, 5))
plt.plot(training_rewards, label="Episode Rewards", color='black', alpha=0.5)
plt.xlabel("Episodes")
plt.ylabel("Episode Rewards")
plt.title("Episode reward evolution")
plt.legend()
plt.savefig("training_rewards.png")  # Save the figure

# Test the model
test_rewards = []
episode_rewards = 0
obs, _ = env.reset()
test_episode_length = 10
for _ in range(test_episode_length):
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_rewards += reward
        if done:
            test_rewards.append(episode_rewards)
            episode_rewards = 0
            obs, _ = env.reset()
        env.render()

# Plot testing rewards
plt.figure(figsize=(10, 5))
plt.plot(test_rewards, label="Cumulative Testing Rewards")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Testing Cumulative Rewards Over Episodes")
plt.legend()
plt.savefig("testing_rewards.png")  # Save the figure
