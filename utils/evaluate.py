# utils/evaluate.py

import gymnasium as gym
import numpy as np
from tqdm import tqdm

def evaluate(agent, env_name="LunarLander-v3", num_episodes=10, render=True):
    """
    评估训练好的 DQN 智能体
    :param agent: 训练好的 DQN agent
    :param env_name: 环境名称
    :param num_episodes: 测试回合数
    :param render: 是否渲染画面
    """
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    rewards = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(obs)  # 注意：此时 epsilon 应为 0（或改为 greedy）
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"Evaluation finished. Average reward over {num_episodes} episodes: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, std_reward