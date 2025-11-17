"""
Random Actions Demo
Demonstrates the environment with random actions (no learning)
"""

import sys
import time
from custom_env import PrivFedFraudEnv


def main():
    """Run random actions demo with visualization"""
    
    print("=" * 60)
    print("PrivFed Random Actions Demo")
    print("=" * 60)
    print("\nThis demo shows random action selection (no learning)")
    print("Close the pygame window to exit\n")
    
    # Create environment with rendering
    env = PrivFedFraudEnv(render_mode='human', max_steps=50)
    
    # Run episode
    observation, info = env.reset()
    
    episode_reward = 0
    action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL_REVIEW"}
    
    print(f"{'Step':<6} {'Action':<15} {'Reward':<10} {'Total':<10}")
    print("-" * 50)
    
    done = False
    step = 0
    
    while not done:
        # Render environment
        env.render()
        
        # Take random action
        action = env.action_space.sample()
        
        # Set action for display
        if env.renderer:
            env.renderer.set_action(action)
        
        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        step += 1
        
        # Print step info
        print(f"{step:<6} {action_names[action]:<15} {reward:<10.2f} {episode_reward:<10.2f}")
        
        # Slow down for visualization
        time.sleep(0.5)
    
    print("-" * 50)
    print(f"\nEpisode finished!")
    print(f"Total Steps: {step}")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Fraud Caught: {info['fraud_caught']}")
    print(f"Fraud Missed: {info['fraud_missed']}")
    print(f"Correct Decisions: {info['correct_decisions']}")
    
    # Keep window open
    print("\nClose the pygame window to exit...")
    try:
        while True:
            env.render()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    env.close()


if __name__ == "__main__":
    main()
