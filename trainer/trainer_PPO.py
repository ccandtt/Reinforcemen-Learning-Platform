import os
import numpy as np
import cv2
from PySide6.QtCore import Signal, QObject

class TrainerSignals(QObject):
    env_frame_ready = Signal(np.ndarray)
    plot_frame_ready = Signal(np.ndarray)
    progress_updated = Signal(int, int)  # current_episode, total_episodes

class TrainerPPO:
    def __init__(self, env, agent, episodes=500, max_steps=500, save_path="result/ppo",
                 use_ensure=False, state_preprocess=None,render_sleep=None):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps = max_steps
        self.save_path = save_path
        self.use_ensure = use_ensure
        self.state_preprocess = state_preprocess
        self.render_sleep = render_sleep  # ★ 新增
        self.reward_history = []
        self.step_used = []
        # 新增：记录loss和策略分布（概率）
        self.loss_history = []
        self.policy_dists = []

        # 创建信号对象
        self.signals = TrainerSignals()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def train(self,reward_type,env_name):
        prev_episode_plot = None  # 缓存上一次绘制好的 reward/loss 折线图
        for episode in range(1, self.episodes + 1):
            state, _ = self.env.reset() if isinstance(self.env.reset(), tuple) else (self.env.reset(), None)
            if self.state_preprocess:
                state = self.state_preprocess(state)
            total_reward = 0

            for step in range(self.max_steps):
                # 渲染环境
                frame = self.env.render()
                if isinstance(frame, np.ndarray):
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.signals.env_frame_ready.emit(frame)
                
                if step == 0:
                    # 更新训练曲线
                    plot_frame = self.draw_reward_loss_curve(frame, self.reward_history, self.loss_history)
                    if plot_frame is not None:
                        self.signals.plot_frame_ready.emit(plot_frame)
                        prev_episode_plot = plot_frame

                # 每一步都更新左侧 frame，并拼接上右侧缓存图（不要覆盖左边）
                if prev_episode_plot is not None:
                    # 只取右边 reward/loss 区域
                    reward_loss_area = prev_episode_plot[:, frame.shape[1]:]  # 提取右边曲线图
                    # frame = np.concatenate([frame, reward_loss_area], axis=1)  # 拼接左边环境和右边图

                if self.render_sleep:  # ★ 新增
                    import time
                    time.sleep(self.render_sleep)

                action, log_prob, best_action = self.agent.choose_act(state)

                if hasattr(self.env, 'spec') and self.env.spec.id == "CartPole-v0":
                    def action_to_arrow(act):
                        return "<-" if act == 0 else "->"

                    actual_arrow = action_to_arrow(action)
                    best_arrow = action_to_arrow(best_action)

                    # 绘制文字和箭头，颜色已交换：
                    # Actual 是蓝色，Best 是红色
                    cv2.putText(frame, f"Next Action: {actual_arrow}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)  # 蓝色
                    cv2.putText(frame, f"Best Action: {best_arrow}", (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)  # 红色

                # ✅ 仅限迷宫用 ensure 可视化贪婪动作
                if self.use_ensure:
                    self.env.ensure(best_action)
                    self.env.define(reward_type)

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, done, truncated, info = step_result
                    done = done or truncated
                else:
                    next_state, reward, done, info = step_result

                if self.state_preprocess:
                    next_state = self.state_preprocess(next_state)

                self.agent.store_transition(state, action, reward, log_prob, done)
                state = next_state
                total_reward += reward

                if done or step == self.max_steps - 1:
                    loss, policy_probs = self.agent.update(return_metrics=True)  # 需要在agent.update里返回loss和策略概率

                    # 记录loss和策略分布
                    self.loss_history.append(loss)
                    self.policy_dists.append(policy_probs)
                    print(f"[PPO] Episode {episode}/{self.episodes} - Reward: {total_reward} - Steps: {step + 1}")
                    self.step_used.append(step + 1)
                    if not env_name == "CartPole-v0":
                        # self.env.maze_view.update_reward_plot(episode, total_reward)
                        # self.env.maze_view.update_loss_plot(episode, loss)
                        pass
                    break

            self.reward_history.append(total_reward)
            # 更新进度
            self.signals.progress_updated.emit(episode, self.episodes)

        if hasattr(self.env, 'spec') and self.env.spec.id == "CartPole-v0":
            cv2.destroyAllWindows()

    def draw_reward_loss_curve(self, frame, reward_history, loss_history, width=800):
        """绘制训练曲线
        
        Args:
            frame: 环境渲染帧
            reward_history: 奖励历史
            loss_history: 损失历史
            width: 图表宽度
        """
        height = 400  # 增加高度以提高清晰度
        
        # 创建纯白背景的图像
        plot_frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 如果数据点太少，返回空白图
        if len(reward_history) < 2:
            return plot_frame

        # 设置边距
        left_margin = 80
        right_margin = 80
        top_margin = 50
        bottom_margin = 50
        plot_width = width - left_margin - right_margin
        plot_height = height - top_margin - bottom_margin

        # 获取数据范围
        reward_min = min(reward_history)
        reward_max = max(reward_history)
        loss_min = min(loss_history) if loss_history else 0
        loss_max = max(loss_history) if loss_history else 1

        # 添加一些边界空间
        reward_range = reward_max - reward_min
        loss_range = loss_max - loss_min
        reward_min -= reward_range * 0.1
        reward_max += reward_range * 0.1
        loss_min -= loss_range * 0.1
        loss_max += loss_range * 0.1

        # 绘制网格
        for i in range(6):
            # 水平网格线
            y = top_margin + i * plot_height // 5
            cv2.line(plot_frame, (left_margin, y), (width - right_margin, y), (220, 220, 220), 1)
            
            # 垂直网格线
            x = left_margin + i * plot_width // 5
            cv2.line(plot_frame, (x, top_margin), (x, height - bottom_margin), (220, 220, 220), 1)

        # 绘制坐标轴
        cv2.line(plot_frame, (left_margin, height - bottom_margin), (width - right_margin, height - bottom_margin), (0, 0, 0), 2)  # X轴
        cv2.line(plot_frame, (left_margin, top_margin), (left_margin, height - bottom_margin), (0, 0, 0), 2)  # Y轴

        # 映射函数
        def map_x(episode_idx):
            return int(left_margin + (episode_idx / (len(reward_history) - 1)) * plot_width)

        def map_reward_y(reward):
            return int(height - bottom_margin - ((reward - reward_min) / (reward_max - reward_min)) * plot_height)

        def map_loss_y(loss):
            return int(height - bottom_margin - ((loss - loss_min) / (loss_max - loss_min)) * plot_height)

        # 绘制reward曲线
        points = [(map_x(i), map_reward_y(r)) for i, r in enumerate(reward_history)]
        for i in range(len(points) - 1):
            cv2.line(plot_frame, points[i], points[i + 1], (0, 180, 0), 2)

        # 绘制loss曲线
        if loss_history:
            points = [(map_x(i), map_loss_y(l)) for i, l in enumerate(loss_history)]
            for i in range(len(points) - 1):
                cv2.line(plot_frame, points[i], points[i + 1], (180, 0, 0), 2)

        # 绘制Y轴刻度（左侧reward）
        for i in range(6):
            y = top_margin + i * plot_height // 5
            value = reward_max - i * (reward_max - reward_min) / 5
            cv2.putText(plot_frame, f"{value:.1f}", (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)

        # 绘制Y轴刻度（右侧loss）
        for i in range(6):
            y = top_margin + i * plot_height // 5
            value = loss_max - i * (loss_max - loss_min) / 5
            text = f"{value:.1f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(plot_frame, text, (width - right_margin + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 1)

        # 绘制X轴刻度
        for i in range(6):
            x = left_margin + i * plot_width // 5
            episode = i * (len(reward_history) - 1) // 5
            text = str(episode)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(plot_frame, text, (x - text_size[0]//2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 添加标题和图例
        cv2.putText(plot_frame, "Training Progress", (width//2 - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # 图例
        legend_x = width - right_margin - 150
        cv2.putText(plot_frame, "Reward", (legend_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        cv2.putText(plot_frame, "Loss", (legend_x + 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 0), 2)

        # 坐标轴标签
        cv2.putText(plot_frame, "Episode", (width//2 - 30, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(plot_frame, "Value", (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return plot_frame
