"""
REINFORCE Algorithm - Manual Implementation
Policy Gradient method with Monte Carlo returns
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.custom_env import PrivFedFraudEnv


class PolicyNetwork(nn.Module):
    """Neural network for policy"""
    
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class REINFORCEAgent:
    """REINFORCE algorithm implementation"""
    
    def __init__(self, obs_dim, action_dim, learning_rate=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Storage for episode
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        """Select action from policy"""
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()
    
    def compute_returns(self):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def update_policy(self):
        """Update policy using REINFORCE"""
        returns = self.compute_returns()
        
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode storage
        self.log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model"""
        self.policy.load_state_dict(torch.load(filepath))


def train_reinforce(config, run_id):
    """Train REINFORCE agent"""
    
    print(f"\n{'='*60}")
    print(f"Training REINFORCE Run {run_id}")
    print(f"{'='*60}")
    print(f"Config: {config}")
    
    env = PrivFedFraudEnv(max_steps=100)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCEAgent(
        obs_dim, 
        action_dim,
        learning_rate=config['learning_rate'],
        gamma=config['gamma']
    )
    
    num_episodes = config['num_episodes']
    episode_rewards = []
    
    # Training loop
    pbar = tqdm(range(num_episodes), desc=f"REINFORCE Run {run_id}")
    for episode in pbar:
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Collect episode
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            episode_reward += reward
            done = terminated or truncated
        
        # Update policy
        loss = agent.update_policy()
        episode_rewards.append(episode_reward)
        
        # Update progress bar
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_postfix({'avg_reward': f'{avg_reward:.2f}'})
    
    # Save model
    os.makedirs("models/pg", exist_ok=True)
    agent.save(f"models/pg/reinforce_run_{run_id}.pth")
    
    # Evaluate
    results = evaluate_reinforce(agent, env, run_id, config)
    
    return results, episode_rewards


def evaluate_reinforce(agent, env, run_id, config):
    """Evaluate REINFORCE agent"""
    
    eval_rewards = []
    eval_fraud_caught = []
    eval_fraud_missed = []
    eval_steps = []
    
    for _ in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            state = torch.FloatTensor(obs)
            with torch.no_grad():
                probs = agent.policy(state)
                action = torch.argmax(probs).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        
        eval_rewards.append(episode_reward)
        eval_fraud_caught.append(info['fraud_caught'])
        eval_fraud_missed.append(info['fraud_missed'])
        eval_steps.append(steps)
    
    results = {
        'algorithm': 'REINFORCE',
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
    """Run REINFORCE hyperparameter search"""
    
    print("=" * 60)
    print("REINFORCE Training - Manual Implementation")
    print("=" * 60)
    
    # Hyperparameter configurations
    configs = [
        # Baseline
        {'learning_rate': 1e-3, 'gamma': 0.99, 'num_episodes': 500},
        # Lower learning rate
        {'learning_rate': 5e-4, 'gamma': 0.99, 'num_episodes': 500},
        # Higher learning rate
        {'learning_rate': 5e-3, 'gamma': 0.99, 'num_episodes': 500},
        # Higher gamma
        {'learning_rate': 1e-3, 'gamma': 0.995, 'num_episodes': 500},
        # Lower gamma
        {'learning_rate': 1e-3, 'gamma': 0.95, 'num_episodes': 500},
        # More episodes
        {'learning_rate': 1e-3, 'gamma': 0.99, 'num_episodes': 1000},
        # Conservative
        {'learning_rate': 1e-4, 'gamma': 0.99, 'num_episodes': 500},
        # Aggressive
        {'learning_rate': 1e-2, 'gamma': 0.98, 'num_episodes': 500},
        # Balanced 1
        {'learning_rate': 3e-4, 'gamma': 0.99, 'num_episodes': 750},
        # Balanced 2
        {'learning_rate': 7e-4, 'gamma': 0.995, 'num_episodes': 600},
    ]
    
    # Train all configurations
    all_results = []
    all_curves = []
    
    for i, config in enumerate(configs):
        results, curve = train_reinforce(config, i+1)
        all_results.append(results)
        all_curves.append(curve)
    
    # Save results
    os.makedirs("results/training_logs", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/training_logs/reinforce_results.csv", index=False)
    
    # Plot comparison
    plot_results(all_results, all_curves)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete - Summary")
    print("=" * 60)
    print("\nTop 3 Configurations:")
    top_3 = results_df.nlargest(3, 'mean_reward')
    print(top_3[['run_id', 'mean_reward', 'std_reward', 'learning_rate', 'gamma']])


def plot_results(all_results, all_curves):
    """Plot training results"""
    
    os.makedirs("results/figures", exist_ok=True)
    
    # Mean reward comparison
    plt.figure(figsize=(12, 6))
    run_ids = [r['run_id'] for r in all_results]
    mean_rewards = [r['mean_reward'] for r in all_results]
    std_rewards = [r['std_reward'] for r in all_results]
    
    plt.bar(run_ids, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color='mediumseagreen')
    plt.xlabel('Run ID', fontsize=12)
    plt.ylabel('Mean Evaluation Reward', fontsize=12)
    plt.title('REINFORCE: Hyperparameter Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/reinforce_comparison.png', dpi=300)
    plt.close()
    
    # Training curves for top 3
    top_indices = sorted(range(len(mean_rewards)), key=lambda i: mean_rewards[i], reverse=True)[:3]
    
    plt.figure(figsize=(12, 6))
    for idx in top_indices:
        curve = all_curves[idx]
        smoothed = pd.Series(curve).rolling(window=10, min_periods=1).mean()
        plt.plot(smoothed, label=f'Run {run_ids[idx]} (LR={all_results[idx]["learning_rate"]:.0e})', alpha=0.8)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('REINFORCE: Training Curves (Top 3)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/reinforce_curves.png', dpi=300)
    plt.close()
    
    print("Plots saved to results/figures/")


if __name__ == "__main__":
    main()
