import gym
import numpy as np
import random
import os
from collections import deque

class SARSA():
    def __init__(self, state_list, action_size, learning_rate, reward_decay, e_greedy):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = reward_decay
        # e-贪婪
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = e_greedy
        self.alpha = learning_rate
        self.q = self.init_q()

    def init_q(self):
        q = {}
        for i in self.state_list:
            q[i] = {}
            for j in range(self.action_size):
                q[i][j] = 0
        return q

    # 动作选择策略
    def choose_act(self, state):
        # e-贪婪：不断减小 epsilon 值，加速收敛
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        val = -10000000
        best_action = 0
        choose_action = None
        for i in range(self.action_size):
            if self.q[state][i] > val:
                best_action = i
                val = self.q[state][i]
        # e-贪婪：有 epsilon 的概率选择随机操作
        if random.random() < self.epsilon:
            choose_action = np.random.randint(self.action_size)
        else:
            choose_action = best_action

        ####
        return choose_action, best_action

    # 更新 Q 表
    def update_q(self, state, action, reward, state_, action_):
        self.q[state][action] = self.q[state][action] + \
                                self.alpha * (reward + self.gamma * self.q[state_][action_] - self.q[state][action])




########
class NStepSARSA:
    def __init__(self, state_list, action_size, learning_rate, reward_decay, e_greedy, n_step=3):
        self.state_list = state_list
        self.action_size = action_size
        self.gamma = reward_decay
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = e_greedy
        self.alpha = learning_rate
        self.n = n_step

        self.q = self.init_q()

        # 用于存储 n 步轨迹（s, a, r）
        self.state_buffer = deque()
        self.action_buffer = deque()
        self.reward_buffer = deque()

    def init_q(self):
        q = {}
        for state in self.state_list:
            q[state] = {}
            for action in range(self.action_size):
                q[state][action] = 0
        return q

    def choose_act(self, state):
        # e-贪婪：不断减小 epsilon 值，加速收敛
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        val = -10000000
        best_action = 0
        choose_action = None
        for i in range(self.action_size):
            if self.q[state][i] > val:
                best_action = i
                val = self.q[state][i]
        # e-贪婪：有 epsilon 的概率选择随机操作
        if random.random() < self.epsilon:
            choose_action = np.random.randint(self.action_size)
        else:
            choose_action = best_action

        ####
        return choose_action, best_action

    def store_transition(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

        # 保持最多 n 个
        if len(self.reward_buffer) > self.n:
            self.state_buffer.popleft()
            self.action_buffer.popleft()
            self.reward_buffer.popleft()

    def update_q(self, next_state, next_action, done):
        """
        next_state, next_action: 下一状态和动作
        done: episode 是否结束
        """
        if len(self.reward_buffer) < self.n:
            return  # 轨迹不足 n 步时不更新

        # 计算 n 步目标 G
        G = 0
        for i in range(self.n):
            G += (self.gamma ** i) * self.reward_buffer[i]

        if not done:
            G += (self.gamma ** self.n) * self.q[next_state][next_action]

        # 更新 Q 值：从最早那个 (s, a) 开始更新
        s = self.state_buffer[0]
        a = self.action_buffer[0]
        self.q[s][a] += self.alpha * (G - self.q[s][a])

        # 移除最旧一项，使缓冲区可以移动
        self.state_buffer.popleft()
        self.action_buffer.popleft()
        self.reward_buffer.popleft()

    def reset_buffers(self):
        self.state_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()