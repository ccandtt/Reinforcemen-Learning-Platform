import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import imageio
import io


def plot_training_progress(rewards: List[float],
                         epsilons: List[float],
                         window_size: int = 10,
                         figsize: Tuple[int, int] = (10, 5),
                         save_path: Optional[str] = None,
                         show: bool = True) -> Optional[np.ndarray]:
    """
    Plot training progress including rewards and epsilon values.
    
    Args:
        rewards: List of episode rewards
        epsilons: List of epsilon values
        window_size: Window size for moving average
        figsize: Figure size
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Plot as numpy array if show is False, else None
    """
    # Calculate moving averages
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        moving_avg = rewards
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, 'b-', alpha=0.3, label='Episode Reward')
    plt.plot(np.arange(len(moving_avg)) + window_size-1, moving_avg, 'r-', label=f'{window_size}-Episode Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.legend()
    
    # Plot epsilon
    plt.subplot(1, 2, 2)
    plt.plot(epsilons, 'g-', label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate')
    plt.grid(True)
    plt.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    if not show:
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        plt.close()
        return img
    else:
        plt.show()
        plt.close()
        return None


def create_training_gif(frames: List[np.ndarray],
                       output_path: str,
                       fps: int = 30,
                       quality: int = 8):
    """
    Create GIF from training frames.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Path to save the GIF
        fps: Frames per second
        quality: GIF quality (1-10, higher is better)
    """
    # Ensure frames are in correct format (uint8)
    frames = [frame.astype(np.uint8) for frame in frames]
    
    # Calculate duration between frames
    duration = 1.0 / fps
    
    # Save GIF
    imageio.mimsave(
        output_path,
        frames,
        duration=duration,
        loop=0,
        quality=quality
    )


def plot_value_function(q_table: dict,
                       maze_size: Tuple[int, int],
                       save_path: Optional[str] = None,
                       show: bool = True) -> Optional[np.ndarray]:
    """
    Plot the value function (maximum Q-value for each state).
    
    Args:
        q_table: Q-table dictionary
        maze_size: Size of the maze (height, width)
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Plot as numpy array if show is False, else None
    """
    # Create value matrix
    value_matrix = np.zeros(maze_size)
    
    # Fill value matrix with maximum Q-values
    for state, q_values in q_table.items():
        if isinstance(q_values, np.ndarray):
            value_matrix[state] = np.max(q_values)
        else:
            value_matrix[state] = max(q_values.values())
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot value function
    plt.imshow(value_matrix, cmap='viridis')
    plt.colorbar(label='Maximum Q-Value')
    
    # Add grid
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.xticks(np.arange(-.5, maze_size[1], 1), [])
    plt.yticks(np.arange(-.5, maze_size[0], 1), [])
    
    # Add labels
    plt.title('Value Function')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    if not show:
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        plt.close()
        return img
    else:
        plt.show()
        plt.close()
        return None


def plot_policy(q_table: dict,
                maze_size: Tuple[int, int],
                save_path: Optional[str] = None,
                show: bool = True) -> Optional[np.ndarray]:
    """
    Plot the learned policy (best action for each state).
    
    Args:
        q_table: Q-table dictionary
        maze_size: Size of the maze (height, width)
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Plot as numpy array if show is False, else None
    """
    # Create policy matrix
    policy_matrix = np.zeros(maze_size, dtype=int)
    
    # Action symbols
    action_symbols = ['↑', '→', '↓', '←']
    
    # Fill policy matrix with best actions
    for state, q_values in q_table.items():
        if isinstance(q_values, np.ndarray):
            policy_matrix[state] = np.argmax(q_values)
        else:
            policy_matrix[state] = max(q_values.items(), key=lambda x: x[1])[0]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot policy
    for i in range(maze_size[0]):
        for j in range(maze_size[1]):
            plt.text(j, i, action_symbols[policy_matrix[i, j]],
                    ha='center', va='center')
    
    # Add grid
    plt.grid(True, which='major', color='black', linestyle='-', linewidth=1)
    plt.imshow(np.zeros(maze_size), cmap='binary')
    
    # Add labels
    plt.title('Learned Policy')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    
    if not show:
        # Convert plot to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        plt.close()
        return img
    else:
        plt.show()
        plt.close()
        return None 