"""
🎮 Game-Like Fraud Detection Demo
Demonstrates the new arcade-style visualization with animated orbs, gates, and effects.
"""

import sys
import time
from custom_env import PrivFedFraudEnv

try:
    from game_renderer import GameRenderer
except ImportError:
    try:
        from environment.game_renderer import GameRenderer
    except ImportError:
        GameRenderer = None

def main():
    """Run demo with game-like visualization"""
    
    print("=" * 60)
    print("🎮 Fraud Detection Arena - Game-Like Visualization")
    print("=" * 60)
    print("\nWatch the RL agent in action with:")
    print("  • Animated transaction orbs")
    print("  • Action gates (Approve/Block/Review)")
    print("  • Consequence animations")
    print("  • Agent brain avatar")
    print("  • Real-time probability bars")
    print("  • Threat meter & timeline")
    print("\nControls:")
    print("  ESC - Exit")
    print("  F3  - Toggle debug info")
    print("  M   - Toggle sound (if enabled)")
    print("\n" + "=" * 60)
    
    # Create environment with game renderer
    env = PrivFedFraudEnv(render_mode='human', max_steps=50)
    
    # Reset environment
    observation, info = env.reset()
    
    # Force initialize game renderer
    if GameRenderer is not None:
        try:
            env.renderer = GameRenderer(env)
            print("✓ Game renderer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize game renderer: {e}")
            print("Falling back to default renderer...")
            env.renderer = None
    else:
        print("✗ Game renderer not available")
        print("Falling back to default renderer...")
        env.renderer = None
    
    # Run episode with random actions (for demo)
    # In real usage, use trained model predictions
    action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL_REVIEW"}
    
    print("\nStarting episode...")
    print("Close the pygame window or press ESC to exit\n")
    
    done = False
    step = 0
    
    try:
        while not done:
            # Render with game renderer
            quit_signal = env.render(use_game_renderer=True)
            if quit_signal:
                break
            
            # Get action (random for demo, replace with model.predict() for trained agent)
            action = env.action_space.sample()
            
            # Set action probabilities (for visualization)
            # In real usage, get these from model
            import numpy as np
            probs = np.random.dirichlet([1, 1, 1])  # Random probabilities for demo
            action_probs = {0: float(probs[0]), 1: float(probs[1]), 2: float(probs[2])}
            
            # Set action in renderer if it's the game renderer
            if GameRenderer is not None and env.renderer and isinstance(env.renderer, GameRenderer):
                env.renderer.set_action(action, action_probs)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Print step info (optional, can be removed for cleaner demo)
            if step % 10 == 0:
                print(f"Step {step}: {action_names[action]}, Reward: {reward:.2f}, "
                      f"Fraud Caught: {info.get('fraud_caught', 0)}, "
                      f"Missed: {info.get('fraud_missed', 0)}")
            
            # Control speed
            time.sleep(0.05)  # Adjust for desired speed
        
        print("\n" + "=" * 60)
        print("Episode Complete!")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {info.get('cumulative_reward', 0):.2f}")
        print(f"Fraud Caught: {info.get('fraud_caught', 0)}")
        print(f"Fraud Missed: {info.get('fraud_missed', 0)}")
        print(f"Correct Decisions: {info.get('correct_decisions', 0)}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        env.close()
        print("\nDemo closed. Thanks for watching! 🎮")


if __name__ == "__main__":
    main()

