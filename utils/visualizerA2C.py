# 文件路径：utils/visualizer_A2C.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_metrics(trainer, agent, save_path="result/a2c"):
    os.makedirs(save_path, exist_ok=True)

    # --- 奖励曲线 ---
    plt.figure()
    plt.plot(trainer.reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "reward_curve.png"))
    plt.close()

    # --- 损失曲线 ---
    if hasattr(trainer, "loss_history"):
        plt.figure()
        plt.plot(trainer.loss_history, label="Critic Loss", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "loss_curve.png"))
        plt.close()

    # --- 策略随 step 变化图 ---
    if hasattr(trainer, "policy_history") and trainer.policy_history:
        # 收集所有 step 的策略概率
        all_policies = []
        for episode_policy in trainer.policy_history:
            if episode_policy is not None:
                all_policies.extend(episode_policy)

        if all_policies:
            all_policies = np.array(all_policies)  # shape: [total_steps, action_dim]

            plt.figure(figsize=(10, 6))
            for i in range(all_policies.shape[1]):
                plt.plot(all_policies[:, i], label=f"Action {i}")
            plt.xlabel("Time Step")
            plt.ylabel("Action Probability")
            plt.title("Policy Evolution Over Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_path, "policy_evolution.png"))
            plt.close()
