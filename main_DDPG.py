import gym
import numpy as np
from algorithm.DDPG import DDPG
from trainer.trainer_DDPG import TrainerDDPG
from utils.loggerDDPG import generate_experiment_report_pdf  # 复用PPO的logger
def main_DDPG(config, return_trainer=False):
    env_name = config['env_name']
    reward_type = "time_penalty"
    # 创建环境
    if env_name == "Pendulum-v1":
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        gamma=config["reward_decay"],
        tau=0.005,
        lr=config["learning_rate"],
        use_nstep_td=config['td_target'],
        n_step=config['n_step'],
        use_per=config["use_per"],
        hide_n1=config['hide_n1'],
        hide_list1= config['hide_list1'],  # type: list
        hide_n2=config['hide_n2'],
        hide_list2= config['hide_list2']   # type: list
    )

    trainer = TrainerDDPG(
        env=env,
        agent=agent,
        episodes=config["episode"],
        state_preprocess=lambda s: np.array(s, dtype=np.float32),
        render_sleep=0.03
    )
    # 返回plot_frame
    if return_trainer:
        return trainer
    
    trainer.train()
    generate_experiment_report_pdf(trainer)


