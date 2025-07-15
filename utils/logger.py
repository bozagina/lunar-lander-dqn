import matplotlib.pyplot as plt
from collections import deque
import os
import numpy as np
class RewardLogger:
    def __init__(self, save_path="logs/q_curve.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.save_path = save_path
        self.rewards = []
        self.avg100 = deque(maxlen=100)

    def log(self, episode_reward):
        self.rewards.append(episode_reward)
        self.avg100.append(episode_reward)

    def plot(self):
        plt.clf()
        plt.title("Q-Learning on LunarLander-v3")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(self.rewards, alpha=0.4, label="episode reward")
        if len(self.avg100) > 0:
            # 计算滑动平均
            avg = [np.mean(self.rewards[max(0, i-99):i+1]) for i in range(len(self.rewards))]
            plt.plot(avg, color="red", label="avg-100")
        plt.legend()
        plt.pause(0.001)  # 非阻塞刷新
        plt.savefig(self.save_path)

    def close(self):
        plt.ioff()
        plt.show()