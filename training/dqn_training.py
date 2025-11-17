"""
DQN Training with Hyperparameter Search
Trains Deep Q-Network on PrivFedFraudEnv with multiple configurations
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import PrivFedFraudEnv


class MetricsCallback(BaseCallback):
    """Custom callback to log training metrics"""
    
    def __init__(self, eval_freq=1000, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        
    def _on_step(self):
        # Log episode completion
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        return True


def train_dqn(config, run_id):
    """Train a single DQN model with given configuration"""
    
    print(f"\n{'='*60}")
    print(f"Training DQN Run {run_id}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    # Create environment
    env = PrivFedFraudEnv(max_steps=100)
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        buffer_size=config['buffer_size'],
        learning_starts=config['learning_starts'],
        target_update_interval=config['target_update_interval'],
        exploration_fraction=config['exploration_fraction'],
        exploration_final_eps=config['exploration_final_eps'],
        verbose=0,
        tensorboard_log=f"./results/tensorboard/dqn_run_{run_id}/"
    )
    
    # Create callback
    callback = MetricsCallback(eval_freq=1000)
    
    # Train
    total_timesteps = config['total_timesteps']
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Save model
    os.makedirs("models/dqn", exist_ok=True)
    model.save(f"models/dqn/dqn_run_{run_id}")
    
    # Evaluate
    eval_env = PrivFedFraudEnv(max_steps=100)
    eval_rewards = []
    eval_fraud_caught = []
    eval_fraud_missed = []
    
    for _ in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        eval_rewards.append(episode_reward)
        eval_fraud_caught.append(info['fraud_caught'])
        eval_fraud_missed.append(info['fraud_missed'])
    
    # Calculate metrics
    results = {
        'run_id': run_id,
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'mean_fraud_caught': np.mean(eval_fraud_caught),
        'mean_fraud_missed': np.mean(eval_fraud_missed),
        **config
    }
    
    print(f"\nResults:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Fraud Caught: {results['mean_fraud_caught']:.1f}")
    print(f"  Fraud Missed: {results['mean_fraud_missed']:.1f}")
    
    return results, callback.episode_rewards


def main():
    """Run DQN hyperparameter search"""
    
    print("=" * 60)
    print("DQN Training - Hyperparameter Search")
    print("=" * 60)
    
    # Define hyperparameter configurations
    configs = [
        # Baseline
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Lower learning rate
        {
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Higher gamma (more long-term focus)
        {
            'learning_rate': 1e-3,
            'gamma': 0.995,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Larger batch size
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 64,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Larger buffer
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 50000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # More exploration
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.3,
            'exploration_final_eps': 0.1,
            'total_timesteps': 50000
        },
        # Faster target updates
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 250,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Early learning start
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 500,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
        # Conservative exploration
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'batch_size': 32,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.05,
            'exploration_final_eps': 0.02,
            'total_timesteps': 50000
        },
        # High learning rate + small batch
        {
            'learning_rate': 5e-3,
            'gamma': 0.99,
            'batch_size': 16,
            'buffer_size': 10000,
            'learning_starts': 1000,
            'target_update_interval': 500,
            'exploration_fraction': 0.1,
            'exploration_final_eps': 0.05,
            'total_timesteps': 50000
        },
    ]
    
    # Train all configurations
    all_results = []
    all_training_curves = []
    
    for i, config in enumerate(configs):
        results, training_curve = train_dqn(config, i+1)
        all_results.append(results)
        all_training_curves.append(training_curve)
    
    # Save results
    os.makedirs("results/training_logs", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/training_logs/dqn_results.csv", index=False)
    
    # Plot comparison
    plot_results(all_results, all_training_curves)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    print("\nTop 3 Configurations by Mean Reward:")
    top_3 = results_df.nlargest(3, 'mean_reward')
    print(top_3[['run_id', 'mean_reward', 'std_reward', 'learning_rate', 'gamma', 'batch_size']])
    
    print(f"\nAll results saved to: results/training_logs/dqn_results.csv")
    print(f"Best model saved as: models/dqn/dqn_run_{top_3.iloc[0]['run_id']}")


def plot_results(all_results, all_training_curves):
    """Plot training results"""
    
    os.makedirs("results/figures", exist_ok=True)
    
    # Plot 1: Mean reward comparison
    plt.figure(figsize=(12, 6))
    run_ids = [r['run_id'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    std_rewards = [r['std_reward'] for r in all_results]
    
    plt.bar(run_ids, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('Mean Evaluation Reward', fontsize=12)
    plt.title('DQN: Hyperparameter Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/dqn_comparison.png', dpi=300)
    plt.close()
    
    print("Plots saved to results/figures/")


if __name__ == "__main__":
    main()
