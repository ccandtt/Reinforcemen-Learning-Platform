import gym
import gym_maze1
import numpy as np
from algorithm.A2C import A2C
from trainer.trainer_A2C import TrainerA2C
from utils.loggerA2C import generate_experiment_report_pdf
from config import ENV_CONFIG
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
reward_type = "time_penalty"


def main_A2C(config, return_trainer=False):
    # 环境的选择性
    if 'maze_size' in config.keys():
        env_name=f"maze-random-{config['maze_size']}x{config['maze_size']}-v0"
        maze_size = config['maze_size']
    else:
        env_name=config['env_name']
    if env_name == "CartPole-v0":
        env = gym.make(env_name, render_mode="rgb_array")
    # 创建环境
    else :
        env = gym.make(env_name)
    # 探索率获取
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Agent初始化（use_nstep_td 控制用TD还是GAE）
    agent = A2C(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=config['reward_decay'], # 折扣因子
        lr=config["learning_rate"], # 学习率
        n_step=config['n_step'], # 如果使用td_target
        use_td_target=config['td_target'], # 是否使用td？
        gae_lambda=0.95,
        hide_n=config['hide_n'],
        hide_list=config['hide_list']
    )
    # use_ensure,区分两环境
    if env_name == "CartPole-v0":
        config['use_ensure']=False
    else:
        config['use_ensure']=True
    # Trainer初始化（use_ensure 控制环境展示best动作等可视化）
    trainer = TrainerA2C(
        env=env,
        agent=agent,
        episodes=config['episode'],
        use_ensure=config['use_ensure'],
        state_preprocess=lambda s: np.array(s, dtype=np.float32),
        render_sleep=0.03
    )

    if return_trainer:
        return trainer

    # 开始训练
    trainer.train(reward_type=reward_type, env_name=env_name)
    from utils.visualizerA2C import plot_training_metrics  # 顶部导入
    plot_training_metrics(trainer, agent, trainer.save_path)
    # 生成实验报告
    generate_experiment_report_pdf(trainer)
