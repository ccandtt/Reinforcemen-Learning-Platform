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


class PPO:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=3e-4, update_steps=10,
                 use_nstep_td=False, n_step=3,hide_n=128,hide_list=[1]):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        self.use_nstep_td = use_nstep_td  # ★ 新增
        self.n_step = n_step
        # ★ 新增
        self.entropy_coef = 0.05  # 可以试0.02 ~ 0.05

        self.policy = ActorCritic(state_dim, action_dim,hide_n,hide_list)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.old_policy = ActorCritic(state_dim, action_dim,hide_n,hide_list)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

        self.loss_history = []
        self.policy_dists = []

    def choose_act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 找出 best 动作（最大概率）
        best_action = torch.argmax(dist.probs).item()

        return action.item(), log_prob.detach(), best_action

    def store_transition(self, state, action, reward, log_prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob.detach())  # 一定detach!
        self.dones.append(done)

    def get_values(self, states):
        states = torch.FloatTensor(np.array(states, dtype=np.float32))
        _, values = self.policy(states)
        return values.squeeze()

    def compute_returns(self):
        if not self.use_nstep_td:
            # ✅ 保持你原来全回报的方式
            returns = []
            G = 0
            for r, d in zip(reversed(self.rewards), reversed(self.dones)):
                if d:
                    G = 0
                G = r + self.gamma * G
                returns.insert(0, G)
            return torch.tensor(returns, dtype=torch.float32)

        else:
            # ✅ 新增 n步TD target
            returns = []
            trajectory_len = len(self.rewards)
            values = self.get_values(self.states).detach().cpu().numpy()
            for t in range(trajectory_len):
                G = 0
                for k in range(self.n_step):
                    if t + k < trajectory_len:
                        G += (self.gamma ** k) * self.rewards[t + k]
                        if self.dones[t + k]:
                            break
                if t + self.n_step < trajectory_len and not self.dones[t + self.n_step - 1]:
                    G += (self.gamma ** self.n_step) * values[t + self.n_step]
                returns.append(G)
            return torch.tensor(returns, dtype=torch.float32)

    def update(self, return_metrics=False):
        returns = self.compute_returns()
        states = torch.FloatTensor(np.array(self.states, dtype=np.float32))
        actions = torch.tensor(self.actions)
        old_log_probs = torch.stack(self.log_probs)

        with torch.no_grad():
            _, values = self.policy(states)
            advantages = returns - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_steps):
            logits, values = self.policy(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.loss_history.append(loss.item())
        with torch.no_grad():
            logits, _ = self.policy(states)
            dist = Categorical(logits=logits)
            probs = dist.probs.mean(dim=0).cpu().numpy()
            self.policy_dists.append(probs)

        # 清空 buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.dones.clear()

        if return_metrics:
            return loss.item(), probs
