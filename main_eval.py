# main_eval.py

from algorithms.dqn import DQNAgent
from utils.evaluate import evaluate
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_space = env.action_space

    # 创建 agent 并加载模型
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,action_space=action_space)
    agent.load("models/dqn_best.pth")  # 加载保存的模型
    agent.set_eval()  # 设置为评估模式

    # 开始评估
    evaluate(agent, env_name="LunarLander-v3", num_episodes=10, render=True)