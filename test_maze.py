import gym
import gym_maze
import time

def test_maze():
    # Create environment
    env = gym.make('Maze-v0', maze_size=(5, 5), render_mode="human")
    
    # Reset environment
    state, _ = env.reset()
    print(f"Initial state: {state}")
    
    # Run a few random steps
    for i in range(10):
        # Take random action
        action = env.action_space.sample()
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Next state: {next_state}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")
        
        # Render and wait
        env.render()
        time.sleep(0.5)
        
        if done:
            print("Goal reached!")
            break
    
    # Close environment
    env.close()

if __name__ == "__main__":
    test_maze() 