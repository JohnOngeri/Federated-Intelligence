"""
Policy Gradient Training (PPO and A2C)
Trains both PPO and A2C algorithms with hyperparameter search
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
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
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        return True


def train_ppo(config, run_id):
    """Train a single PPO model"""
    
    print(f"\n{'='*60}")
    print(f"Training PPO Run {run_id}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    env = PrivFedFraudEnv(max_steps=100)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        ent_coef=config['ent_coef'],
        clip_range=config['clip_range'],
        verbose=0,
        tensorboard_log=f"./results/tensorboard/ppo_run_{run_id}/"
    )
    
    callback = MetricsCallback()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    
    # Save model
    os.makedirs("models/pg", exist_ok=True)
    model.save(f"models/pg/ppo_run_{run_id}")
    
    # Evaluate
    results = evaluate_model(model, run_id, 'PPO', config)
    
    return results, callback.episode_rewards


def train_a2c(config, run_id):
    """Train a single A2C model"""
    
    print(f"\n{'='*60}")
    print(f"Training A2C Run {run_id}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    env = PrivFedFraudEnv(max_steps=100)
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        n_steps=config['n_steps'],
        ent_coef=config['ent_coef'],
        vf_coef=config['vf_coef'],
        verbose=0,
        tensorboard_log=f"./results/tensorboard/a2c_run_{run_id}/"
    )
    
    callback = MetricsCallback()
    model.learn(total_timesteps=config['total_timesteps'], callback=callback, progress_bar=True)
    
    # Save model
    os.makedirs("models/pg", exist_ok=True)
    model.save(f"models/pg/a2c_run_{run_id}")
    
    # Evaluate
    results = evaluate_model(model, run_id, 'A2C', config)
    
    return results, callback.episode_rewards


def evaluate_model(model, run_id, algorithm, config):
    """Evaluate trained model"""
    
    eval_env = PrivFedFraudEnv(max_steps=100)
    eval_rewards = []
    eval_fraud_caught = []
    eval_fraud_missed = []
    eval_steps = []
    
    for _ in range(10):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        eval_rewards.append(episode_reward)
        eval_fraud_caught.append(info['fraud_caught'])
        eval_fraud_missed.append(info['fraud_missed'])
        eval_steps.append(steps)
    
    results = {
        'algorithm': algorithm,
        'run_id': run_id,
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'mean_fraud_caught': np.mean(eval_fraud_caught),
        'mean_fraud_missed': np.mean(eval_fraud_missed),
        'mean_steps': np.mean(eval_steps),
        **config
    }
    
    print(f"\nResults:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Fraud Caught: {results['mean_fraud_caught']:.1f}")
    print(f"  Fraud Missed: {results['mean_fraud_missed']:.1f}")
    
    return results


def main():
    """Run PPO and A2C hyperparameter search"""
    
    print("=" * 60)
    print("Policy Gradient Training - PPO and A2C")
    print("=" * 60)
    
    # PPO Configurations
    ppo_configs = [
        # Baseline
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # Higher learning rate
        {'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # More steps per update
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 4096, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # Larger batch
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 128, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # More epochs
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 20, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # Higher entropy
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.05, 'clip_range': 0.2, 'total_timesteps': 50000},
        # Tighter clip
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.1, 'total_timesteps': 50000},
        # Higher gamma
        {'learning_rate': 3e-4, 'gamma': 0.995, 'n_steps': 2048, 'batch_size': 64, 
         'n_epochs': 10, 'ent_coef': 0.01, 'clip_range': 0.2, 'total_timesteps': 50000},
        # Conservative
        {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 1024, 'batch_size': 32, 
         'n_epochs': 5, 'ent_coef': 0.001, 'clip_range': 0.15, 'total_timesteps': 50000},
        # Aggressive
        {'learning_rate': 5e-3, 'gamma': 0.98, 'n_steps': 512, 'batch_size': 128, 
         'n_epochs': 15, 'ent_coef': 0.1, 'clip_range': 0.3, 'total_timesteps': 50000},
    ]
    
    # A2C Configurations
    a2c_configs = [
        # Baseline
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # Higher learning rate
        {'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # More steps
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 10, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # Higher entropy
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.05, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # Higher value coefficient
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 5, 'ent_coef': 0.01, 
         'vf_coef': 1.0, 'total_timesteps': 50000},
        # Higher gamma
        {'learning_rate': 7e-4, 'gamma': 0.995, 'n_steps': 5, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # Larger rollout
        {'learning_rate': 7e-4, 'gamma': 0.99, 'n_steps': 20, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
        # Conservative
        {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 3, 'ent_coef': 0.001, 
         'vf_coef': 0.25, 'total_timesteps': 50000},
        # Aggressive
        {'learning_rate': 5e-3, 'gamma': 0.98, 'n_steps': 10, 'ent_coef': 0.1, 
         'vf_coef': 0.75, 'total_timesteps': 50000},
        # Balanced
        {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 8, 'ent_coef': 0.02, 
         'vf_coef': 0.5, 'total_timesteps': 50000},
    ]
    
    # Train PPO
    print("\n" + "=" * 60)
    print("Training PPO Models")
    print("=" * 60)
    ppo_results = []
    ppo_curves = []
    for i, config in enumerate(ppo_configs):
        results, curve = train_ppo(config, i+1)
        ppo_results.append(results)
        ppo_curves.append(curve)
    
    # Train A2C
    print("\n" + "=" * 60)
    print("Training A2C Models")
    print("=" * 60)
    a2c_results = []
    a2c_curves = []
    for i, config in enumerate(a2c_configs):
        results, curve = train_a2c(config, i+1)
        a2c_results.append(results)
        a2c_curves.append(curve)
    
    # Save results
    os.makedirs("results/training_logs", exist_ok=True)
    ppo_df = pd.DataFrame(ppo_results)
    a2c_df = pd.DataFrame(a2c_results)
    ppo_df.to_csv("results/training_logs/ppo_results.csv", index=False)
    a2c_df.to_csv("results/training_logs/a2c_results.csv", index=False)
    
    # Plot comparison
    plot_results(ppo_results, a2c_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    print("\nTop 3 PPO Configurations:")
    print(ppo_df.nlargest(3, 'mean_reward')[['run_id', 'mean_reward', 'std_reward', 'learning_rate', 'n_steps']])
    print("\nTop 3 A2C Configurations:")
    print(a2c_df.nlargest(3, 'mean_reward')[['run_id', 'mean_reward', 'std_reward', 'learning_rate', 'n_steps']])


def plot_results(ppo_results, a2c_results):
    """Plot training results"""
    
    os.makedirs("results/figures", exist_ok=True)
    
    # Combined comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # PPO
    ppo_ids = [r['run_id'] for r in ppo_results]
    ppo_rewards = [r['mean_reward'] for r in ppo_results]
    ppo_stds = [r['std_reward'] for r in ppo_results]
    ax1.bar(ppo_ids, ppo_rewards, yerr=ppo_stds, capsize=5, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Run ID', fontsize=11)
    ax1.set_ylabel('Mean Reward', fontsize=11)
    ax1.set_title('PPO: Hyperparameter Comparison', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # A2C
    a2c_ids = [r['run_id'] for r in a2c_results]
    a2c_rewards = [r['mean_reward'] for r in a2c_results]
    a2c_stds = [r['std_reward'] for r in a2c_results]
    ax2.bar(a2c_ids, a2c_rewards, yerr=a2c_stds, capsize=5, alpha=0.7, color='coral')
    ax2.set_xlabel('Run ID', fontsize=11)
    ax2.set_ylabel('Mean Reward', fontsize=11)
    ax2.set_title('A2C: Hyperparameter Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/pg_comparison.png', dpi=300)
    plt.close()
    
    print("Plots saved to results/figures/")


if __name__ == "__main__":
    main()
