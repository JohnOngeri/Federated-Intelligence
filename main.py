"""
Main Demonstration Script
Loads the best-performing model and runs it with full visualization
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
import torch

from environment.custom_env import PrivFedFraudEnv
from training.reinforce_training import PolicyNetwork


def load_best_model():
    """Identify and load the best-performing model across all algorithms"""
    
    print("=" * 60)
    print("Loading Best Model")
    print("=" * 60)
    
    # Load all results
    dqn_results = pd.read_csv("results/training_logs/dqn_results.csv")
    ppo_results = pd.read_csv("results/training_logs/ppo_results.csv")
    a2c_results = pd.read_csv("results/training_logs/a2c_results.csv")
    reinforce_results = pd.read_csv("results/training_logs/reinforce_results.csv")
    
    # Find best from each algorithm
    best_dqn = dqn_results.loc[dqn_results['mean_reward'].idxmax()]
    best_ppo = ppo_results.loc[ppo_results['mean_reward'].idxmax()]
    best_a2c = a2c_results.loc[a2c_results['mean_reward'].idxmax()]
    best_reinforce = reinforce_results.loc[reinforce_results['mean_reward'].idxmax()]
    
    # Compare all
    candidates = {
        'DQN': (best_dqn['mean_reward'], best_dqn['run_id'], 'dqn'),
        'PPO': (best_ppo['mean_reward'], best_ppo['run_id'], 'ppo'),
        'A2C': (best_a2c['mean_reward'], best_a2c['run_id'], 'a2c'),
        'REINFORCE': (best_reinforce['mean_reward'], best_reinforce['run_id'], 'reinforce')
    }
    
    # Find overall best
    best_algorithm = max(candidates.items(), key=lambda x: x[1][0])
    algo_name, (reward, run_id, model_type) = best_algorithm
    
    print(f"\nBest Algorithm: {algo_name}")
    print(f"  Run ID: {int(run_id)}")
    print(f"  Mean Reward: {reward:.2f}")
    print(f"\nAll Algorithm Performance:")
    for name, (rew, rid, _) in candidates.items():
        print(f"  {name:12} Run {int(rid):2d}: {rew:7.2f}")
    
    # Load the best model
    print(f"\nLoading {algo_name} model...")
    
    if model_type == 'dqn':
        model = DQN.load(f"models/dqn/dqn_run_{int(run_id)}")
        model_info = ('DQN', 'Stable Baselines3')
    elif model_type == 'ppo':
        model = PPO.load(f"models/pg/ppo_run_{int(run_id)}")
        model_info = ('PPO', 'Stable Baselines3')
    elif model_type == 'a2c':
        model = A2C.load(f"models/pg/a2c_run_{int(run_id)}")
        model_info = ('A2C', 'Stable Baselines3')
    else:  # REINFORCE
        env_temp = PrivFedFraudEnv()
        obs_dim = env_temp.observation_space.shape[0]
        action_dim = env_temp.action_space.n
        policy = PolicyNetwork(obs_dim, action_dim)
        policy.load_state_dict(torch.load(f"models/pg/reinforce_run_{int(run_id)}.pth"))
        model = policy
        model_info = ('REINFORCE', 'Custom PyTorch')
    
    return model, algo_name, model_info, int(run_id)


def run_demonstration(model, algorithm_name, model_info, run_id, num_episodes=3):
    """Run the model with full visualization"""
    
    print("\n" + "=" * 60)
    print("Starting Demonstration")
    print("=" * 60)
    print(f"Algorithm: {model_info[0]} ({model_info[1]})")
    print(f"Number of Episodes: {num_episodes}")
    print("\nClose the pygame window to stop the demonstration")
    print("=" * 60)
    
    # Create environment with rendering
    env = PrivFedFraudEnv(render_mode='human', max_steps=100)
    
    action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL_REVIEW"}
    
    all_episode_rewards = []
    
    try:
        for episode in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1} / {num_episodes}")
            print(f"{'='*60}")
            
            obs, info = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            print(f"{'Step':<6} {'Bank':<8} {'Amount':<12} {'Action':<15} {'Reward':<10} {'Total':<10}")
            print("-" * 70)
            
            while not done:
                # Render
                env.render()
                
                # Get action from model
                if algorithm_name == 'REINFORCE':
                    state = torch.FloatTensor(obs)
                    with torch.no_grad():
                        probs = model(state)
                        action = torch.argmax(probs).item()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                # Display action
                if env.renderer:
                    env.renderer.set_action(action)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step += 1
                
                # Print step information
                bank_id = info.get('current_bank', -1)
                print(f"{step:<6} Bank {bank_id+1:<4} "
                      f"${env.current_transaction['amount']:<10.2f} "
                      f"{action_names[action]:<15} {reward:<10.2f} {episode_reward:<10.2f}")
                
                obs = next_obs
                
                # Slow down for visualization
                time.sleep(0.3)
            
            print("-" * 70)
            print(f"Episode {episode + 1} Complete:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps Taken: {step}")
            print(f"  Fraud Caught: {info['fraud_caught']}")
            print(f"  Fraud Missed: {info['fraud_missed']}")
            print(f"  Correct Decisions: {info['correct_decisions']}")
            
            all_episode_rewards.append(episode_reward)
            
            # Brief pause between episodes
            if episode < num_episodes - 1:
                print("\nStarting next episode in 2 seconds...")
                time.sleep(2)
        
        # Final statistics
        print("\n" + "=" * 60)
        print("Demonstration Complete - Final Statistics")
        print("=" * 60)
        print(f"Average Reward: {np.mean(all_episode_rewards):.2f}")
        print(f"Std Reward: {np.std(all_episode_rewards):.2f}")
        print(f"Best Episode: {np.max(all_episode_rewards):.2f}")
        print(f"Worst Episode: {np.min(all_episode_rewards):.2f}")
        
        # Keep window open
        print("\nClose the pygame window to exit...")
        while True:
            env.render()
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    
    finally:
        env.close()


def main():
    """Main execution"""
    
    # Check if models exist
    if not os.path.exists("results/training_logs/dqn_results.csv"):
        print("ERROR: No training results found!")
        print("Please run training scripts first:")
        print("  python training/dqn_training.py")
        print("  python training/pg_training.py")
        print("  python training/reinforce_training.py")
        return
    
    # Load best model
    model, algorithm_name, model_info, run_id = load_best_model()
    
    # Run demonstration
    run_demonstration(model, algorithm_name, model_info, run_id, num_episodes=3)


if __name__ == "__main__":
    main()
