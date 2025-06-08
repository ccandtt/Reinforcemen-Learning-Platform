#####SARSA算法支持的是
import gym
from algorithm.Sarsa import SARSA
from algorithm.Sarsa import NStepSARSA
from trainer.trainer_SARSA import  Trainer
from utils.logger import generate_experiment_report_pdf
import gym_maze1


def main_SARSA(config,return_trainer=False):
    env_name = f"maze-random-{config['maze_size']}x{config['maze_size']}-v0"
    algorithm = config["algorithm"]
    # 学习率
    learning_rate = config["learning_rate"]
    # 折扣因子
    reward_decay = config["reward_decay"]
    # 探索率
    e_greedy = config["e_greedy"]
    # 奖励函数
    reward_type = "time_penalty"
    # bool变量，是否支持TD taget
    td_target = config["td_target"]
    maze_size = config['maze_size']
    action_num = 4
    env = gym.make(env_name)
    state_list = []
    # 初始化 Q 表
    for i in range(maze_size):
        for j in range(maze_size):
            state_list.append((i, j))
    if td_target:
        n_step = config["n_step"]
        agent = NStepSARSA(state_list, action_num, learning_rate, reward_decay, e_greedy, n_step=n_step)
    else:
        agent = SARSA(state_list, action_num, learning_rate, reward_decay, e_greedy)
    trainer = Trainer(env, agent, episodes=config['episode'], save_path=f"result/{algorithm}")
    if return_trainer:
        return trainer
    trainer.train(reward_type, td_target)
    generate_experiment_report_pdf(trainer)
