# 这里是传入所有参数的接口
# 你需要传入参数，
import numpy as np
from main_SARSA import main_SARSA
from main_DQN import main_DQN
from main_A2C import main_A2C
from main_DDPG import main_DDPG
from main_PPO import main_PPO

# 下面是这些算法共有的参数，字典形式
# config的参数定义
config = {}

# DDPG+自己的环境
config = {
    "env_name": "Pendulum-v1",  # DDPG智能返回此环境
    "reward_decay": 0.99,  # 折扣因子
    "learning_rate": 3e-4,  # 学习率
    "algorithm": 'DDPG',  # 算法名称
    'hide_n1': 3,  # 隐藏层个数
    'hide_list1': [128, 128, 128],  # 隐藏层各层神经元个数,
    'hide_n2': 3,  # 隐藏层个数
    'hide_list2': [128, 128, 128],  # 隐藏层各层神经元个数,
    'episode': 100,
    'td_target': True,  # FALSE则是1
    'n_step': 3,
    "use_per": True,
}

# 你需要传入的参数实例：SARSA+迷宫
# config = {
#     "maze_size":5,
#     "reward_decay": 0.95, # 折扣因子
#     "learning_rate": 0.5,# 学习率
#     "e_greedy": 0.03,  # 探索率
#     "algorithm":'SARSA', #算法名称
#     'td_target':True,
#     'n_step':3,
#     'episode':100
# }
# DQN+迷宫, DQN换成倒立摆需要'env_name'参数
# config = {
#     "maze_size":5,
#     "reward_decay": 0.95, # 折扣因子
#     "learning_rate": 0.5,# 学习率
#     "e_greedy": 0.03,  # 探索率
#     "algorithm":'DQN', #算法名称
#     'hide_n':1,  #隐藏层个数
#     'hide_list':[4],#隐藏层各层神经元个数
#     'use_replay':False, #是否使用经验回放？,
# 'episode':100
# }


# A2C+迷宫
# config = {
#     "maze_size": 5,
#     "reward_decay": 0.99,  # 折扣因子
#     "learning_rate": 3e-4,  # 学习率
#     "algorithm": 'A2C',  #算法名称
#     'hide_n': 1,  #隐藏层个数
#     'hide_list': [128],  #隐藏层各层神经元个数,
#     'episode': 100,
#     'td_target': True,
#     'n_step': 3
# }


# PPO+迷宫
# config = {
#     "maze_size": 5,
#     "reward_decay": 0.99,  # 折扣因子
#     "learning_rate": 3e-4,  # 学习率
#     "algorithm": 'PPO',  #算法名称
#     'hide_n': 1,  #隐藏层个数
#     'hide_list': [128],  #隐藏层各层神经元个数,
#     'episode': 100,
#     'td_target': True,
#     'n_step': 3
# }
# PPO+CartPole-v0
# config = {
#     "env_name": "CartPole-v0",
#     "reward_decay": 0.99,  # 折扣因子
#     "learning_rate": 3e-4,  # 学习率
#     "algorithm": 'PPO',  # 算法名称
#     'hide_n': 1,  # 隐藏层个数
#     'hide_list': [128],  # 隐藏层各层神经元个数,
#     'episode': 100,
#     'td_target': True,
#     'n_step': 3
# }
# DQN+CartPole-v0
# config = {
#     "env_name":"CartPole-v0",
#     "reward_decay": 0.95, # 折扣因子
#     "learning_rate": 0.5,# 学习率
#     "e_greedy": 0.03,  # 探索率
#     "algorithm":'DQN', #算法名称
#     'hide_n':2,  #隐藏层个数
#     'hide_list':[4,4],#隐藏层各层神经元个数
#     'use_replay':False, #是否使用经验回放？,
# 'episode':100
# }
# A2C+CartPole-v0
# config = {
#     "env_name": "CartPole-v0",
#     "reward_decay": 0.99,  # 折扣因子
#     "learning_rate": 3e-4,  # 学习率
#     "algorithm": 'A2C',  #算法名称
#     'hide_n': 1,  #隐藏层个数
#     'hide_list': [128],  #隐藏层各层神经元个数,
#     'episode': 100,
#     'td_target': True,
#     'n_step': 3
# }
if __name__ == "__main__":
    if config['algorithm'] == 'SARSA':
        main_SARSA(config)
    elif config['algorithm'] == 'DQN':
        main_DQN(config)
    elif config['algorithm'] == 'A2C':
        main_A2C(config)
    elif config['algorithm'] == 'DDPG':
        main_DDPG(config)
    elif config['algorithm'] == 'PPO':
        main_PPO(config)
    # 此部分定义传入参数意外的默认参数
    # if config.keys[0] == "CartPole-v0":
    #     config["CartPole-v0"]["render_mode"] = "human"
    #     config["CartPole-v0"]["render_sleep"] = 0.03
    #     config["CartPole-v0"]["use_ensure"] = False
    #     config["CartPole-v0"]["state_preprocess"] = lambda s: np.array(s, dtype=np.float32)



