# main.py

from algorithms.dqn import DQNAgent
from utils.train import train
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    action_space = env.action_space
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim,action_space=action_space)

    train(agent, env_name="LunarLander-v3", episodes=600)