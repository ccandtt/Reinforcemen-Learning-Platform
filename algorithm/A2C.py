import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hide_n, hide_list):
        super(ActorCritic, self).__init__()
        assert hide_n == len(hide_list), "隐藏层数量 hide_n 与 hide_list 长度不一致"
        layers = []
        # 第一层输入 → 第一个隐藏层
        layers.append(nn.Linear(state_dim, hide_list[0]))

        # 中间隐藏层
        for i in range(1, hide_n):
            layers.append(nn.Linear(hide_list[i - 1], hide_list[i]))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        # print(self.fc)
        self.actor = nn.Linear(hide_list[-1], action_dim)
        self.critic = nn.Linear(hide_list[-1], 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)


class A2C:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, n_step=1, use_td_target=False, gae_lambda=0.95,
                 hide_n=1, hide_list=[4]):
        self.gamma = gamma
        self.n_step = n_step
        self.use_td_target = use_td_target
        self.gae_lambda = gae_lambda

        self.policy = ActorCritic(state_dim, action_dim, hide_n, hide_list)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()

        self.memory = []  # n-step缓存

        self.last_loss = None  # 用于可视化记录
        self.last_policy = None  # 用于可视化记录

    def choose_act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        best_action = torch.argmax(dist.probs).item()
        return action.item(), log_prob.detach(), best_action

    def compute_return(self, rewards, next_value, dones):
        R = next_value
        returns = []
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - done)
            returns.insert(0, R)
        return returns

    def compute_gae(self, rewards, values, next_value, dones):
        gae = 0
        returns = []
        values = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            returns.insert(0, gae + values[t])
        return returns

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, done))

        if len(self.memory) < self.n_step and not done:
            return None, None

        states, actions, rewards, dones = zip(*self.memory)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)

        _, next_value = self.policy(torch.FloatTensor(next_state).unsqueeze(0))

        values = [self.policy(torch.FloatTensor(s).unsqueeze(0))[1].item() for s in states]

        if self.use_td_target:
            targets = self.compute_return(rewards, next_value.item(), dones)
        else:
            targets = self.compute_gae(rewards, values, next_value.item(), dones)

        targets = torch.FloatTensor(targets)
        values = torch.FloatTensor(values)

        logits, value_preds = self.policy(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        advantages = targets - value_preds.squeeze()

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = self.MseLoss(value_preds.squeeze(), targets)

        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---- ✅ 可视化记录 ----
        probs = dist.probs.detach().cpu().numpy()
        self.last_loss = loss.item()
        self.last_policy = probs

        self.memory = []  # 清空缓存

        return loss.item(), probs

    def get_last_metrics(self):
        return self.last_loss, self.last_policy
