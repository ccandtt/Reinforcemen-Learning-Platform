class SumTree:
    def __init__(self, capacity):
        # capacity：叶节点数量（最大经验容量），使用 (2*capacity-1) 大小的列表维护二叉树
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)  # 存储优先度和
        self.data = [None] * capacity  # 存储经验元组 (s,a,r,s',done)
        self.size = 0  # 当前存储的经验数量
        self.write = 0  # 下一个写入位置（循环覆盖）

    def add(self, priority, data):
        # 添加新经验及其优先度
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)  # 更新叶节点及其父节点的和

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        # 将节点 idx 的优先度更新为新的 priority，并向上递归更新其父节点的和
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2  # 父节点索引
            self.tree[idx] += change

    def get_leaf(self, value):
        # 根据采样的 value 值从根节点开始遍历，找到对应的叶节点
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):  # 若已到叶节点
                leaf_idx = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        # 返回根节点的值，即所有叶节点优先度之和
        return self.tree[0]


import numpy as np
import random


class MemoryPER:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, eps=0.01):
        self.capacity = capacity  # 最大容量
        self.alpha = alpha  # 优先度指数
        self.beta = beta  # IS权重指数起始值
        self.beta_increment = beta_increment
        self.eps = eps  # 避免优先度为零的小常量
        self.tree = SumTree(capacity)  # SumTree 结构

    def add(self, s, a, r, s_, done):
        # 存储新经验，优先度初始化为当前最大优先度或1
        data = (s, a, r, s_, done)
        max_p = max(self.tree.tree[self.capacity - 1:self.capacity - 1 + self.tree.size], default=1)
        max_p = max(max_p, 1.0)
        self.tree.add(max_p, data)

    def sample(self, n):
        # 采样 n 条经验，返回批量数据及对应的树索引和重要性权重
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_priority / n

        # 动态递增 beta 值（可选），逐步减小偏差
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # 计算采样概率和重要性采样权重
        probs = np.array(priorities) / self.tree.total_priority
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()  # 归一化权重

        # 分解经验元组为批量 (state, action, reward, next_state, done)
        states = np.vstack([d[0] for d in batch])
        actions = np.array([d[1] for d in batch])
        rewards = np.array([d[2] for d in batch])
        next_states = np.vstack([d[3] for d in batch])
        dones = np.array([d[4] for d in batch])
        return states, actions, rewards, next_states, dones, idxs, weights

    def update(self, idxs, td_errors):
        # 根据TD误差更新对应采样经验的优先度
        for idx, td in zip(idxs, td_errors):
            p = (abs(td) + self.eps) ** self.alpha
            self.tree.update(idx, p)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class Network(nn.Module):
    """
    通用 MLP 网络，支持自定义隐藏层数和每层神经元数量。
    """

    def __init__(self,
                 n_features: int,
                 n_actions: int,
                 hide_n: int,
                 hide_list: list):
        super(Network, self).__init__()

        assert hide_n == len(hide_list), "隐藏层数量 hide_n 与 hide_list 长度不一致"

        layers = []

        # 第一层输入 → 第一个隐藏层
        layers.append(nn.Linear(n_features, hide_list[0]))
        layers.append(nn.ReLU())

        # 中间隐藏层
        for i in range(1, hide_n):
            layers.append(nn.Linear(hide_list[i - 1], hide_list[i]))
            layers.append(nn.ReLU())

        # 最后一层隐藏层 → 输出层
        layers.append(nn.Linear(hide_list[-1], n_actions))

        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)



class DeepQNetworkReplay:
    def __init__(self, n_actions, n_features, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.9, replace_target_iter=100,
                 memory_size=1000, batch_size=32, e_greedy_increment=None,
                 hide_n=1, hide_list=None):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size

        # ε-贪婪策略参数：动态增长机制
        self.epsilon = 0.0 if e_greedy_increment is not None else e_greedy
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment

        self.learn_step_counter = 0

        # 使用 MemoryPER 替代原始经验回放
        self.memory = MemoryPER(capacity=memory_size)
        self.cost_his = []

        # 构建网络：隐藏层层数 hide_n 和神经元个数 hide_list
        if hide_list is None:
            hide_list = []
        assert len(hide_list) == hide_n, "hide_list 长度应等于 hide_n"

        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions, hide_n=hide_n,
                                hide_list=hide_list)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions, hide_n=hide_n,
                                  hide_list=hide_list)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss(reduction='none')  # 使用自定义加权损失

    def store_transition(self, s, a, r, s_, done):
        # 存储到优先经验回放内存
        self.memory.add(s, a, r, s_, done)

    def choose_action(self, state):
        # ε-贪婪选择动作
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        q_values = self.eval_net(state_tensor).detach().numpy()
        best_action = np.argmax(q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action=best_action
        return action,best_action

    def learn(self):
        # 定期更新目标网络
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从优先回放内存中采样
        states, actions, rewards, next_states, dones, idxs, weights = \
            self.memory.sample(self.batch_size)

        # 转换为Tensor
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

        # 计算当前Q值和目标Q值
        q_eval = self.eval_net(states).gather(1, actions)  # Q(s,a)
        q_next = self.target_net(next_states).detach()  # Q(s', *)
        q_next_max = q_next.max(dim=1, keepdim=True)[0]
        q_target = rewards + self.gamma * q_next_max * (1 - dones)  # 目标值

        # 计算带权均方误差损失
        loss = (q_eval - q_target).pow(2)
        weighted_loss = weights * loss
        loss_mean = weighted_loss.mean()

        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # 用新TD误差更新对应经验的优先度
        td_errors = (q_eval - q_target).detach().numpy().squeeze()
        self.memory.update(idxs, td_errors)

        # 记录损失，用于绘图
        self.cost_his.append(loss_mean.item())

        # 动态增加ε
        if self.epsilon < self.epsilon_max:
            self.epsilon += self.epsilon_increment or 0.0
        return loss_mean.item()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
