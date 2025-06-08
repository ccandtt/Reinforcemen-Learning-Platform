import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(trainer, agent, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Reward 曲线
    plt.figure()
    plt.plot(trainer.reward_history, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward_curve.png"))
    plt.close()

    # 2. Loss 曲线
    plt.figure()
    plt.plot(trainer.loss_history, label="Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 3. 策略分布变化
    if len(trainer.policy_dists) > 0:
        policy_dists = np.array(trainer.policy_dists)
        plt.figure(figsize=(10, 6))
        for i in range(policy_dists.shape[1]):
            plt.plot(policy_dists[:, i], label=f"Action {i}")
        plt.xlabel("Update Step")
        plt.ylabel("Action Probability")
        plt.title("Policy Distribution Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "policy_distribution.png"))
        plt.close()
