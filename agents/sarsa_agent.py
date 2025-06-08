import numpy as np
from typing import Tuple, Dict, Any
import json
import os

class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent implementation
    with improved features and flexibility.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 td_target: bool = False):
        """
        Initialize SARSA agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Number of possible actions
            learning_rate: Learning rate (alpha)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            td_target: Whether to use TD target in learning
        """
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.td_target = td_target
        
        # Initialize Q-table as a dictionary for sparse state representation
        self.q_table: Dict[Tuple, np.ndarray] = {}
        
        # Training statistics
        self.stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'epsilon_history': []
        }
    
    def get_state_key(self, state: np.ndarray) -> Tuple:
        """Convert state array to hashable tuple for Q-table key."""
        return tuple(state.astype(int))
    
    def get_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            
        Returns:
            Selected action index
        """
        state_key = self.get_state_key(state)
        
        # Initialize state value if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            return np.argmax(self.q_table[state_key])
    
    def learn(self, 
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             next_action: int,
             done: bool) -> float:
        """
        Update Q-value using SARSA update rule.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            next_action: Next action
            done: Whether episode is done
            
        Returns:
            TD error of the update
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Initialize states in Q-table if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # TD Target
        if done:
            target = reward
        else:
            if self.td_target:
                # Use maximum Q-value as target (like Q-learning)
                target = reward + self.gamma * np.max(self.q_table[next_state_key])
            else:
                # Use next action's Q-value as target (SARSA)
                target = reward + self.gamma * self.q_table[next_state_key][next_action]
        
        # Update Q-value
        td_error = target - current_q
        self.q_table[state_key][action] = current_q + self.lr * td_error
        
        # Debug information
        print(f"Debug - Learning info:")
        print(f"  State: {state}, Action: {action}")
        print(f"  Next state: {next_state}, Next action: {next_action}")
        print(f"  Reward: {reward}, Done: {done}")
        print(f"  Current Q: {current_q}")
        print(f"  Target: {target}")
        print(f"  TD error: {td_error}")
        print(f"  New Q: {self.q_table[state_key][action]}")
        print(f"  Epsilon: {self.epsilon}")
        
        return td_error
    
    def update_epsilon(self):
        """Decay epsilon value."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path: str):
        """
        Save agent's Q-table and parameters.
        
        Args:
            path: Directory path to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        # Convert Q-table keys to strings for JSON serialization
        q_table_serializable = {str(k): v.tolist() for k, v in self.q_table.items()}
        
        # Save parameters
        params = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'td_target': self.td_target,
            'stats': self.stats
        }
        
        # Save files
        with open(os.path.join(path, 'q_table.json'), 'w') as f:
            json.dump(q_table_serializable, f)
        with open(os.path.join(path, 'params.json'), 'w') as f:
            json.dump(params, f)
    
    def load(self, path: str):
        """
        Load agent's Q-table and parameters.
        
        Args:
            path: Directory path to load the agent from
        """
        # Load Q-table
        with open(os.path.join(path, 'q_table.json'), 'r') as f:
            q_table_serialized = json.load(f)
            # Convert string keys back to tuples
            self.q_table = {
                tuple(map(int, k.strip('()').split(','))): np.array(v)
                for k, v in q_table_serialized.items()
            }
        
        # Load parameters
        with open(os.path.join(path, 'params.json'), 'r') as f:
            params = json.load(f)
            self.state_size = params['state_size']
            self.action_size = params['action_size']
            self.lr = params['learning_rate']
            self.gamma = params['gamma']
            self.epsilon = params['epsilon']
            self.epsilon_decay = params['epsilon_decay']
            self.epsilon_min = params['epsilon_min']
            self.td_target = params['td_target']
            self.stats = params['stats']
    
    def update_stats(self, episode_reward: float, episode_length: int):
        """Update training statistics."""
        self.stats['episode_rewards'].append(episode_reward)
        self.stats['episode_lengths'].append(episode_length)
        self.stats['epsilon_history'].append(self.epsilon) 