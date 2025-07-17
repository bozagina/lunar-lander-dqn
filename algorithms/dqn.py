# algorithms/dqn.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import torch.nn.functional as F

# 定义经验回放缓冲区的数据结构
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, action_space, lr=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.action_space = action_space

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.evaluate = False

        # 探索策略
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 软更新参数
        self.tau = 0.005

        # 经验回放
        self.memory = ReplayBuffer(capacity=10000)
        self.batch_size = 128

        # 损失函数和训练计数器
        self.criterion = nn.SmoothL1Loss()
        self.train_step = 0
        # self.update_freq = 4  # 每 4 步更新一次目标网络
        self.update_freq = 1 # 每一步都更新一次目标网络
    def set_eval(self):
        """进入评估模式，关闭 epsilon 探索"""
        self.evaluate = True
        self.epsilon = 0.0  # 关闭探索

    def select_action(self, obs):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if not self.evaluate and np.random.rand() < self.epsilon:
            # 0715 将动作选择修改为在action_space范围内随机选择,而不是使用np.random.randint
            return torch.tensor(
                [[self.action_space.sample()]], dtype=torch.long, device=self.device
            )
        else:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.q_net(obs_tensor).argmax().item()

    def update(self, obs, action, reward, next_obs, done):
        self.memory.push(obs, action, reward, next_obs, done)

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).to(self.device)

        q_values = self.q_net(states).gather(1, actions).squeeze()
        
        # Double DQN 更新方式
        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, best_actions).squeeze()
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        # 每隔几步软更新目标网络
        self.train_step += 1
        if self.train_step % self.update_freq == 0:
            self._soft_update()

        return loss.item()

    def _soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )

    def save(self, path="models/dqn.pth"):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path="models/dqn.pth"):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")