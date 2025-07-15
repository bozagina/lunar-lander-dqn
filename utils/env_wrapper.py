# utils/env_wrapper.py
import gymnasium as gym

def make_env(env_name="CartPole-v1"):
    env = gym.make(env_name)
    return env