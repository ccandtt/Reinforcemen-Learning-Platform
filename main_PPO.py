import gym
import gym_maze1  # 注册迷宫
import numpy as np
from algorithm.PPO import PPO
from trainer.trainer_PPO import TrainerPPO
from utils.loggerPPO import generate_experiment_report_pdf
from config import ENV_CONFIG
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main_PPO(config, return_trainer=False):
    reward_type = "time_penalty"
    if 'maze_size' in config.keys():
        env_name = f"maze-random-{config['maze_size']}x{config['maze_size']}-v0"
        maze_size = config['maze_size']
    else:
        env_name = config['env_name']
    if env_name == "CartPole-v0":
        env = gym.make(env_name, render_mode="rgb_array")
    # 创建环境
    else:
        env = gym.make(env_name)

    # PPO Agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim,
                action_dim,
                gamma=config["reward_decay"],
                lr=config['learning_rate'],

                use_nstep_td=config['td_target'],  # get()去config里的，没有的话default就是False
                n_step=config['n_step'],
                hide_n=config['hide_n'],
                hide_list=config['hide_list'])
    if env_name == "CartPole-v0":
        config["use_ensure"] = False
    else:
        config["use_ensure"] = True

    # Trainer
    trainer = TrainerPPO(
        env=env,
        agent=agent,
        episodes=config['episode'],
        use_ensure=config["use_ensure"],
        state_preprocess=lambda s: np.array(s, dtype=np.float32),
        render_sleep=0.03  # ★ 新加的
    )

    if return_trainer:
        return trainer

    trainer.train(reward_type, env_name)
    #绘制最后的回报，loss，策略图像
    from utils.visualizerPPO import plot_training_metrics  # 顶部导入
    plot_training_metrics(trainer, agent, trainer.save_path)

    generate_experiment_report_pdf(trainer)



