import gym
from algorithm.DQN import DeepQNetwork
from algorithm.DQN_replay import DeepQNetworkReplay
from trainer.trainer_DQN import Trainer
from utils.logger import generate_experiment_report_pdf
import gym_maze1

def main_DQN(config, return_trainer=False):
    if 'maze_size' in config.keys():
        env_name = f"maze-random-{config['maze_size']}x{config['maze_size']}-v0"
        maze_size = config['maze_size']
    else:
        env_name = config['env_name']
    algorithm = config["algorithm"]
    # 学习率
    learning_rate = config["learning_rate"]
    # 折扣因子
    reward_decay = config["reward_decay"]
    # 探索率
    e_greedy = config["e_greedy"]
    # 奖励函数
    reward_type = "time_penalty"
    # 隐藏层个数
    hide_n =config["hide_n"]

    # 隐藏层各层神经元个数
    hide_list = config["hide_list"]
    # 是否使用经验回放？
    use_replay = config['use_replay']
    if not (env_name=="CartPole-v0"):
        action_num = 4
        env = gym.make(env_name)
        n_features = 2
    else:
        action_num = 2
        env = gym.make(env_name, render_mode="rgb_array")
        n_features = 4
    if use_replay:
        agent = DeepQNetworkReplay(action_num, n_features,
                                   learning_rate=0.01,
                                   reward_decay=0.9,
                                   e_greedy=0.9,
                                   replace_target_iter=200,
                                   memory_size=2000, hide_n=hide_n, hide_list=hide_list)

    else:
        agent = DeepQNetwork(action_num, n_features,
                             learning_rate=0.01,
                             reward_decay=0.9,
                             e_greedy=0.9,
                             replace_target_iter=200,
                             memory_size=2000, hide_n=hide_n, hide_list=hide_list)
    trainer = Trainer(env, agent, config['episode'], save_path=f"result/{algorithm}")
    
    if return_trainer:
        return trainer
    
    trainer.train_off(env_name, reward_type, use_replay)
    generate_experiment_report_pdf(trainer)


