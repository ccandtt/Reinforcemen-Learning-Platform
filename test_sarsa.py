from main_SARSA import main_SARSA

def test_sarsa():
    # Configuration for testing
    config = {
        'maze_size': 5,
        'episode': 100,  # Smaller number for testing
        'learning_rate': 0.1,
        'reward_decay': 0.95,
        'e_greedy': 0.1,
        'reward_type': 'dense',
        'render_mode': 'human',  # Use 'human' for visualization
        'random_seed': 42
    }
    
    # Run SARSA training
    trainer = main_SARSA(config, return_trainer=True)
    trainer.train()

if __name__ == "__main__":
    test_sarsa() 