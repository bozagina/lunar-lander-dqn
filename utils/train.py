# utils/train.py

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from algorithms.dqn import DQNAgent
import os
import numpy as np
def train(agent, env_name="LunarLander-v3", episodes=600):
    env = gym.make(env_name, render_mode="human")
    writer = SummaryWriter(log_dir=f"runs/dqn_lunarlander_{agent.tau}_{agent.epsilon_decay}")
    
    best_reward = float('-inf')
    model_save_path = "models/dqn_best.pth"

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Clip reward
            reward = np.clip(reward, -1, 1)

            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        # TensorBoard 记录
        writer.add_scalar("Reward/train", total_reward, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)

        print(f"Episode: {episode + 1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

        # 保存最佳模型
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(model_save_path)

    writer.close()
    env.close()