import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_maze1.envs1.maze_view_2d import MazeView2D


class MazeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render.modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    ACTION = ["N", "S", "E", "W"]

    def __init__(self, maze_file=None, maze_size=None, mode=None, render_mode=None):
        self.render_mode = render_mode
        self.viewer = None
        self.best_action = None
        self.reward_type = None

        try:
            if maze_file:
                self.maze_view = MazeView2D(
                    maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                    maze_file_path=maze_file,
                    screen_size=(800, 1000),
                    render_mode=render_mode
                )
            elif maze_size:
                if mode == "plus":
                    has_loops = True
                    num_portals = int(round(min(maze_size) / 3))
                else:
                    has_loops = False
                    num_portals = 0

                self.maze_view = MazeView2D(
                    maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                    maze_size=maze_size,
                    screen_size=(800, 1000),
                    has_loops=has_loops,
                    num_portals=num_portals,
                    render_mode=render_mode
                )
            else:
                raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

            self.maze_size = self.maze_view.maze_size

            # forward or backward in each dimension
            self.action_space = spaces.Discrete(2 * len(self.maze_size))

            # observation is the x, y coordinate of the grid
            low = np.zeros(len(self.maze_size), dtype=np.float32)
            high = np.array(self.maze_size, dtype=np.float32) - np.ones(len(self.maze_size), dtype=np.float32)
            self.observation_space = spaces.Box(low, high, dtype=np.float32)

            # initial condition
            self.state = np.zeros(len(self.maze_size), dtype=np.float32)
            self.steps_beyond_done = None

            # Simulation related variables.
            self._seed()
            self.reset()

            # Just need to initialize the relevant attributes
            self._configure()
        except Exception as e:
            print(f"Error initializing maze environment: {str(e)}")
            raise

    def __del__(self):
        try:
            if hasattr(self, 'maze_view'):
                self.maze_view.quit_game()
        except Exception as e:
            print(f"Error cleaning up maze environment: {str(e)}")

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def ensure(self, best_action):
        """Store the best action for visualization"""
        if isinstance(best_action, int):
            self.best_action = self.ACTION[best_action]
        else:
            self.best_action = best_action

    def define(self, reward_type):
        """定义奖励类型"""
        self.reward_type = reward_type

    def step(self, action):
        """执行一步动作
        
        Args:
            action: 可以是整数(0,1,2,3)或字符串('N','E','S','W')
        """
        # 将数字动作转换为字符串动作
        if isinstance(action, (int, np.integer)):
            action = self.ACTION[action]
            
        # 将数字最佳动作转换为字符串动作
        if self.best_action is not None and isinstance(self.best_action, (int, np.integer)):
            self.best_action = self.ACTION[self.best_action]
            
        # 移动机器人
        self.maze_view.move_robot(action, self.best_action)

        if self.reward_type is None:
            raise ValueError("Reward type must be defined using define() method before stepping the environment")

        reward, done = self.calculate_reward(self.reward_type, self.maze_view.robot, self.maze_view.goal, self.maze_size)
        self.state = self.maze_view.robot
        info = {}

        return self.state, reward, done, False, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
            
        if hasattr(self, 'maze_view'):
            self.maze_view.reset_robot()
            self.state = np.zeros(len(self.maze_size), dtype=np.float32)
            self.steps_beyond_done = None
            return self.state, {}
        else:
            raise RuntimeError("Maze environment not properly initialized")

    def is_game_over(self):
        return self.maze_view.game_over

    def render(self,mode='rgb_array'):
        """
        Render the current state of the environment.
        Returns:
            numpy.ndarray: RGB array of shape (H, W, 3) representing the current frame
        """
        if hasattr(self, 'maze_view'):
            return self.maze_view.update(mode)
        return None

    def close(self):
        if hasattr(self, 'maze_view'):
            self.maze_view.quit_game()

    def calculate_reward(self, reward_type, robot, goal, maze_size):
        if np.array_equal(robot, goal):
                return 1.0, True
        else:
                return -0.1, False
        ########
        """计算奖励"""
        if reward_type == "time_penalty":
            if np.array_equal(robot, goal):
                return 1.0, True
            else:
                return -0.1, False
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")


class MazeEnvSample5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvSample5x5, self).__init__(maze_file="maze2d_5x5.npy")


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom5x5, self).__init__(maze_size=(5, 5), )


class MazeEnvSample10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvSample10x10, self).__init__(maze_file="maze2d_10x10.npy")


class MazeEnvSample30x30(MazeEnv):

    def __init__(self):
        super(MazeEnvSample30x30, self).__init__(maze_file="maze2d_30x30.npy")


class MazeEnvSample50x50(MazeEnv):

    def __init__(self):
        super(MazeEnvSample50x50, self).__init__(maze_file="maze2d_50x50.npy")


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10, self).__init__(maze_size=(10, 10))


class MazeEnvRandom50x50(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom50x50, self).__init__(maze_size=(50, 50))


class MazeEnvSample3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvSample3x3, self).__init__(maze_file="maze2d_3x3.npy")


class MazeEnvRandom3x3(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom3x3, self).__init__(maze_size=(3, 3))


class MazeEnvSample100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvSample100x100, self).__init__(maze_file="maze2d_100x100.npy")


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom100x100, self).__init__(maze_size=(100, 100))


class MazeEnvRandom30x30(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom30x30, self).__init__(maze_size=(30, 30))


class MazeEnvRandom10x10Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10Plus, self).__init__(maze_size=(10, 10), mode="plus")


class MazeEnvRandom20x20Plus(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom20x20Plus, self).__init__(maze_size=(20, 20), mode="plus")


class MazeEnvRandom30x30Plus(MazeEnv):
    def __init__(self):
        super(MazeEnvRandom30x30Plus, self).__init__(maze_size=(30, 30), mode="plus")
