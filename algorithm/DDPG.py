import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hide_n1, hide_list1):
        super(Actor, self).__init__()
        assert hide_n1 == len(hide_list1), "隐藏层数量 hide_n 与 hide_list 长度不一致"
        layers = []
        # 第一层输入 → 第一个隐藏层
        layers.append(nn.Linear(state_dim, hide_list1[0]))

        # 中间隐藏层
        for i in range(1, hide_n1):
            layers.append(nn.Linear(hide_list1[i - 1], hide_list1[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hide_list1[-1], action_dim))
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, state):
        return self.net(state) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hide_n2, hide_list2):
        super(Critic, self).__init__()
        assert hide_n2 == len(hide_list2), "隐藏层数量 hide_n 与 hide_list 长度不一致"
        layers = []
        # 第一层输入 → 第一个隐藏层
        layers.append(nn.Linear(state_dim + action_dim, hide_list2[0]))

        # 中间隐藏层
        for i in range(1, hide_n2):
            layers.append(nn.Linear(hide_list2[i - 1], hide_list2[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hide_list2[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000, n_step=1, gamma=0.99, use_per=False, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        self.use_per = use_per

        if use_per:
            self.priorities = deque(maxlen=capacity)
            self.alpha = alpha
            self.beta = beta
            self.max_priority = 1.0

    def _get_n_step_transition(self):
        R, next_state, done = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if d:
                break
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        return (state, action, R, next_state, done)

    def add(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        transition = self._get_n_step_transition()
        self.buffer.append(transition)

        if self.use_per:
            max_prio = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(max_prio)

    def sample(self, batch_size):
        if self.use_per:
            priorities = np.array(self.priorities, dtype=np.float32).flatten()  # 🧠 Flatten 一定要做！
            probs = priorities ** self.alpha
            probs = probs / (probs.sum() + 1e-8)
            assert probs.ndim == 1, "probs must be 1-dimensional!"
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max() + 1e-8
            batch = [self.buffer[idx] for idx in indices]
            return batch, indices, torch.FloatTensor(weights).unsqueeze(1)
        else:
            indices = np.random.choice(len(self.buffer), batch_size)
            batch = [self.buffer[idx] for idx in indices]
            return batch, None, torch.ones((batch_size, 1))

    def update_priorities(self, indices, td_errors):
        for i, err in zip(indices, td_errors):  # 此时的td_errors是一个numpy数组
            err_value = float(np.abs(err).item())  # 使用.item()将单元素数组转换为标量
            self.priorities[i] = err_value + 1e-6

    def __len__(self):
        return len(self.buffer)


class DDPG:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=1e-3,
                 use_nstep_td=False, n_step=1, use_per=False, hide_n1=3, hide_list1=None, hide_n2=3, hide_list2=None):
        if hide_list2 is None:
            hide_list2 = [128, 128, 128]
        if hide_list1 is None:
            hide_list1 = [128, 128, 128]
        self.actor = Actor(state_dim, action_dim, max_action, hide_n1, hide_list1)
        self.actor_target = Actor(state_dim, action_dim, max_action, hide_n1, hide_list1)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hide_n2, hide_list2)
        self.critic_target = Critic(state_dim, action_dim, hide_n2, hide_list2)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.use_per = use_per

        self.replay_buffer = ReplayBuffer(
            capacity=100000,
            n_step=n_step if use_nstep_td else 1,
            gamma=gamma,
            use_per=use_per
        )

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise_scale * np.random.normal(size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, transition):
        self.replay_buffer.add(transition)

    def update(self):
        """更新网络参数并返回损失值"""
        if len(self.replay_buffer.buffer) < 128:  # 最小批量大小
            return None, None

        # 从经验回放中采样
        if self.use_per:
            transitions, indices, weights = self.replay_buffer.sample(128)
        else:
            transitions = self.replay_buffer.sample(128)
            weights = np.ones(128)
        
        # 解包转换
        state_batch = torch.FloatTensor(np.array([t[0] for t in transitions]))
        action_batch = torch.FloatTensor(np.array([t[1] for t in transitions]))
        reward_batch = torch.FloatTensor(np.array([t[2] for t in transitions]))
        next_state_batch = torch.FloatTensor(np.array([t[3] for t in transitions]))
        done_batch = torch.FloatTensor(np.array([t[4] for t in transitions]))

        # 计算目标Q值
        with torch.no_grad():
            next_action = self.actor_target(next_state_batch)
            target_Q = self.critic_target(next_state_batch, next_action)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q

        # 更新Critic
        current_Q = self.critic(state_batch, action_batch)
        critic_loss = torch.mean(weights * torch.square(current_Q - target_Q))
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor
        actor_loss = -torch.mean(weights * self.critic(state_batch, self.actor(state_batch)))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # 如果使用PER，更新优先级
        if self.use_per:
            td_errors = torch.abs(current_Q - target_Q).detach().cpu().numpy()
            # 确保td_errors是正确的形状
            td_errors = td_errors.flatten()  # 展平数组以确保形状正确
            self.replay_buffer.update_priorities(indices, td_errors)

        return critic_loss.item(), actor_loss.item()

