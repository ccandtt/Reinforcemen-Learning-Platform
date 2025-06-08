from gym.envs.registration import register


register(
    id='maze-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample5x5',
    max_episode_steps=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom5x5',
    max_episode_steps=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample10x10',
    max_episode_steps=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom10x10',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-random-50x50-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom50x50',
    max_episode_steps=10000,
    nondeterministic=True,
)


register(
    id='maze-random-30x30-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom30x30',
    max_episode_steps=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample3x3',
    max_episode_steps=1000,
)

register(
    id='maze-sample-30x30-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample30x30',
    max_episode_steps=1000,
)

register(
    id='maze-sample-50x50-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample50x50',
    max_episode_steps=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom3x3',
    max_episode_steps=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='gym_maze1.envs1:MazeEnvSample100x100',
    max_episode_steps=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom100x100',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom10x10Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom20x20Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='gym_maze1.envs1:MazeEnvRandom30x30Plus',
    max_episode_steps=1000000,
    nondeterministic=True,
)
