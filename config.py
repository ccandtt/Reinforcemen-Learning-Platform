import numpy as np
def load_config(env_name, algo_name):
    config = {
        "env": {
            "name": env_name,
            "max_steps": 200,
            "reward_threshold": 195 if env_name == "cartpole" else None
        },
        "agent": {
            "name": algo_name,
            "gamma": 0.99,
            "alpha": 0.1,
            "epsilon": 0.1,
            "hidden_size": 128,
            "lr": 1e-3
        },
        "train": {
            "episodes": 500,
            "visualize_interval": 10,
            "save_interval": 100
        }
    }
    return config

ENV_CONFIG = {
    "CartPole-v0": {
        "render_mode": "human",
        "use_ensure": False,
        "render_sleep": 0.03,
        "state_preprocess": lambda s: np.array(s, dtype=np.float32),
        "use_nstep_td": False,  # ★ 新增
        "n_step": 3            # ★ 新增
    },
    "Pendulum-v1": {
        "render_mode": "human",
        "use_ensure": False,
        "render_sleep": 0.03,
        "state_preprocess": lambda s: np.array(s, dtype=np.float32),
        "use_nstep_td": True,  # ✔ 启用n步TD
        "n_step": 5,
        "use_per": True  # ✔ 启用优先级经验重放
    },

}



# 参数配置字典
PARAMS_CONFIG = {
    # 场景1：迷宫环境（自定义网格世界）
    "迷宫": {
        "SARSA": {
            "maze_size": {
                "type": "int",
                "value": 5,
                "min": 3,
                "max": 10,
                "step": 1
            },
            "reward_decay": {
                "type": "float",
                "value": 0.95,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "e_greedy": {
                "type": "float",
                "value": 0.03,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "algorithm": {
                "type": "string",
                "value": "SARSA"
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            }
        },
        "DQN": {
            "maze_size": {
                "type": "int",
                "value": 5,
                "min": 3,
                "max": 10,
                "step": 1
            },
            "reward_decay": {
                "type": "float",
                "value": 0.95,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "e_greedy": {
                "type": "float",
                "value": 0.03,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "algorithm": {
                "type": "string",
                "value": "DQN"
            },
            "hide_n": {
                "type": "int",
                "value": 1,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [4]
            },
            "use_replay": {
                "type": "bool",
                "value": False
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            }
        },
        "A2C": {
            "maze_size": {
                "type": "int",
                "value": 5,
                "min": 3,
                "max": 10,
                "step": 1
            },
            "reward_decay": {
                "type": "float",
                "value": 0.99,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 3e-4,
                "min": 1e-6,
                "max": 1e-2,
                "step": 1e-6
            },
            "algorithm": {
                "type": "string",
                "value": "A2C"
            },
            "hide_n": {
                "type": "int",
                "value": 1,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [128]
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            }
        },
        "PPO": {
            "maze_size": {
                "type": "int",
                "value": 5,
                "min": 3,
                "max": 10,
                "step": 1
            },
            "reward_decay": {
                "type": "float",
                "value": 0.99,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 3e-4,
                "min": 1e-6,
                "max": 1e-2,
                "step": 1e-6
            },
            "algorithm": {
                "type": "string",
                "value": "PPO"
            },
            "hide_n": {
                "type": "int",
                "value": 1,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [128]
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            }
        }
    },
    
    # 场景2：CartPole-v0（车杆平衡问题）
    "CartPole-v0": {
        "DQN": {
            "env_name": {
                "type": "string",
                "value": "CartPole-v0"
            },
            "reward_decay": {
                "type": "float",
                "value": 0.95,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "e_greedy": {
                "type": "float",
                "value": 0.03,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "algorithm": {
                "type": "string",
                "value": "DQN"
            },
            "hide_n": {
                "type": "int",
                "value": 2,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [4, 4]
            },
            "use_replay": {
                "type": "bool",
                "value": False
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            }
        },
        "A2C": {
            "env_name": {
                "type": "string",
                "value": "CartPole-v0"
            },
            "reward_decay": {
                "type": "float",
                "value": 0.99,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 3e-4,
                "min": 1e-6,
                "max": 1e-2,
                "step": 1e-6
            },
            "algorithm": {
                "type": "string",
                "value": "A2C"
            },
            "hide_n": {
                "type": "int",
                "value": 1,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [128]
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            }
        },
        "PPO": {
            "env_name": {
                "type": "string",
                "value": "CartPole-v0"
            },
            "reward_decay": {
                "type": "float",
                "value": 0.99,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 3e-4,
                "min": 1e-6,
                "max": 1e-2,
                "step": 1e-6
            },
            "algorithm": {
                "type": "string",
                "value": "PPO"
            },
            "hide_n": {
                "type": "int",
                "value": 1,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list": {
                "type": "list",
                "value": [128]
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            }
        }
    },
    
    # 场景3：Pendulum-v1（摆摆问题，连续控制）
    "Pendulum-v1": {
        "DDPG": {
            "env_name": {
                "type": "string",
                "value": "Pendulum-v1"
            },
            "reward_decay": {
                "type": "float",
                "value": 0.99,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01
            },
            "learning_rate": {
                "type": "float",
                "value": 3e-4,
                "min": 1e-6,
                "max": 1e-2,
                "step": 1e-6
            },
            "algorithm": {
                "type": "string",
                "value": "DDPG"
            },
            # Actor网络结构
            "hide_n1": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list1": {
                "type": "list",
                "value": [128, 128, 128]
            },
            # Critic网络结构
            "hide_n2": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 5,
                "step": 1
            },
            "hide_list2": {
                "type": "list",
                "value": [128, 128, 128]
            },
            "episode": {
                "type": "int",
                "value": 100,
                "min": 10,
                "max": 1000,
                "step": 10
            },
            "td_target": {
                "type": "bool",
                "value": True
            },
            "n_step": {
                "type": "int",
                "value": 3,
                "min": 1,
                "max": 10,
                "step": 1
            },
            "use_per": {
                "type": "bool",
                "value": True
            }
        }
    }
}

