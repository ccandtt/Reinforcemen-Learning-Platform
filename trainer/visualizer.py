import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 设置中文显示（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_rewards(reward_list):
    plt.figure(figsize=(8, 4))
    plt.plot(reward_list, label="Total Reward per Episode", color='blue')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("训练奖励曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_losses(loss_list):
    if all(v == 0 for v in loss_list):
        return  # 忽略全为0的损失列表（如SARSA等tabular方法）
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, label="Average Loss per Episode", color='red')
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("训练损失曲线")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def animate_maze_trajectory(trajectory, maze_size, episode=""):
    fig, ax = plt.subplots()
    grid = np.zeros((maze_size, maze_size))

    def update(frame):
        ax.clear()
        ax.set_title(f"测试轨迹 - 第{frame + 1}步 ({episode})")
        ax.set_xticks(np.arange(-0.5, maze_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.set_xlim(-0.5, maze_size - 0.5)
        ax.set_ylim(-0.5, maze_size - 0.5)
        ax.invert_yaxis()

        # 绘制所有轨迹点
        for i in range(frame + 1):
            y, x = trajectory[i]
            ax.plot(x, y, "bo", markersize=12 if i == frame else 5)

        # 标出起点和终点
        start_y, start_x = trajectory[0]
        end_y, end_x = trajectory[-1]
        ax.plot(start_x, start_y, "go", label="起点", markersize=10)
        ax.plot(end_x, end_y, "ro", label="终点", markersize=10)

        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=400, repeat=False)
    plt.show()
