# algorithms/q_learning.py
import numpy as np

class QLearningAgent:
    def __init__(self, state_bins=20, action_dim=2, lr=0.1, gamma=0.99, epsilon=0.1):
        self.state_bins = state_bins
        self.action_dim = action_dim
        self.q_table = np.zeros((state_bins, state_bins, state_bins, state_bins, action_dim))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
    
    def discretize(self, obs):
        # 将连续状态离散化到0~state_bins-1
        upper_bounds = [4.8, 10, 0.42, 10]
        lower_bounds = [-4.8, -10, -0.42, -10]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(4)]
        return [min(self.state_bins-1, max(0, int(r * self.state_bins))) for r in ratios]

    def select_action(self, obs):
        state = self.discretize(obs)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        return np.argmax(self.q_table[state[0], state[1], state[2], state[3]])
    
    def update(self, obs, action, reward, next_obs, done):
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)
        best_next = np.max(self.q_table[next_state[0], next_state[1], next_state[2], next_state[3]])
        td_target = reward + self.gamma * best_next * (not done)
        self.q_table[state[0], state[1], state[2], state[3], action] += \
            self.lr * (td_target - self.q_table[state[0], state[1], state[2], state[3], action])