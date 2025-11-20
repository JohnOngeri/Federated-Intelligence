"""
🎮 Game-Like Fraud Detection Demo with Trained Model
Shows the trained RL agent making decisions with the game visualization.
"""

import sys
import time
import os
import numpy as np
from custom_env import PrivFedFraudEnv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from game_renderer import GameRenderer
except ImportError:
    try:
        from environment.game_renderer import GameRenderer
    except ImportError:
        GameRenderer = None

try:
    from stable_baselines3 import DQN, PPO, A2C
    import torch
    from training.reinforce_training import PolicyNetwork
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


def load_best_model():
    """Load the best trained model"""
    import pandas as pd
    
    # Get project root (parent of environment directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Find results directory (check both locations)
    results_paths = [
        os.path.join(project_root, "scripts/results/training_logs/a2c_results.csv"),
        os.path.join(project_root, "results/training_logs/a2c_results.csv"),
        "scripts/results/training_logs/a2c_results.csv",
        "results/training_logs/a2c_results.csv",
    ]
    
    a2c_results_path = None
    for path in results_paths:
        if os.path.exists(path):
            a2c_results_path = path
            break
    
    if not a2c_results_path:
        print("Could not find a2c_results.csv in any expected location")
        return None, None
    
    try:
        a2c_results = pd.read_csv(a2c_results_path)
        best_run = a2c_results.loc[a2c_results['mean_reward'].idxmax()]
        run_id = int(best_run['run_id'])
        
        # Try to load model (check multiple locations)
        model_paths = [
            os.path.join(project_root, f"scripts/models/pg/a2c_run_{run_id}.zip"),
            os.path.join(project_root, f"scripts/models/pg/a2c_run_{run_id}"),
            os.path.join(project_root, f"models/pg/a2c_run_{run_id}.zip"),
            os.path.join(project_root, f"models/pg/a2c_run_{run_id}"),
            f"scripts/models/pg/a2c_run_{run_id}.zip",
            f"scripts/models/pg/a2c_run_{run_id}",
            f"models/pg/a2c_run_{run_id}.zip",
            f"models/pg/a2c_run_{run_id}",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            model = A2C.load(model_path)
            print(f"✓ Loaded A2C model from run {run_id} (reward: {best_run['mean_reward']:.2f})")
            return model, "A2C"
        else:
            print(f"Model file not found for run {run_id}")
            print(f"Checked paths: {model_paths[:4]}...")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None


def get_action_probabilities(model, observation, algorithm_name):
    """Extract action probabilities from model"""
    # For Stable-Baselines3, we can't easily get probabilities
    # So we'll use a simple heuristic based on action selection
    # In practice, you'd need to modify the model or use a custom policy
    
    # For now, return uniform distribution
    # A better approach would be to use model's policy network directly
    return {0: 0.33, 1: 0.33, 2: 0.34}


def main():
    """Run demo with trained model"""
    
    print("=" * 60)
    print("🎮 Fraud Detection Arena - Trained Model Demo")
    print("=" * 60)
    
    # Try to load trained model
    model = None
    algorithm_name = None
    
    if MODELS_AVAILABLE:
        model, algorithm_name = load_best_model()
        if model:
            print(f"Using trained {algorithm_name} model")
        else:
            print("No trained model found, using random actions")
    else:
        print("Model libraries not available, using random actions")
    
    # Create environment with larger budgets for longer demo
    env = PrivFedFraudEnv(render_mode='human', max_steps=8000)
    # Override budget limits for demo
    env.initial_privacy_budget = 10000.0
    env.initial_manual_budget = 2000
    observation, info = env.reset()
    
    # Initialize game renderer
    if GameRenderer is not None:
        try:
            env.renderer = GameRenderer(env)
            print("✓ Game renderer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize game renderer: {e}")
            env.renderer = None
    else:
        print("✗ Game renderer not available")
        env.renderer = None
    
    action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL_REVIEW"}
    
    print("\nStarting episode...")
    print("Close the pygame window or press ESC to exit\n")
    
    done = False
    step = 0
    
    try:
        while not done:
            # Render
            quit_signal = env.render(use_game_renderer=True)
            if quit_signal:
                break
            
            # Get action from model or random
            if model:
                if algorithm_name == "A2C" or algorithm_name == "PPO":
                    action, _ = model.predict(observation, deterministic=False)
                else:
                    action, _ = model.predict(observation, deterministic=True)
                
                # Convert action to integer (handle numpy array)
                if isinstance(action, np.ndarray):
                    action = int(action.item() if action.size == 1 else action[0])
                else:
                    action = int(action)
                
                # Get probabilities (simplified - in practice, extract from policy)
                action_probs = get_action_probabilities(model, observation, algorithm_name)
                # Make selected action more likely
                action_probs[action] = 0.7
                # Normalize
                total = sum(action_probs.values())
                action_probs = {k: v/total for k, v in action_probs.items()}
            else:
                # Random action for demo
                action = env.action_space.sample()
                probs = np.random.dirichlet([1, 1, 1])
                action_probs = {0: float(probs[0]), 1: float(probs[1]), 2: float(probs[2])}
            
            # Set action in renderer
            if GameRenderer is not None and env.renderer and isinstance(env.renderer, GameRenderer):
                env.renderer.set_action(action, action_probs)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Print progress
            if step % 20 == 0:
                print(f"Step {step}: {action_names[action]}, Reward: {reward:.2f}, "
                      f"Fraud Caught: {info.get('fraud_caught', 0)}, "
                      f"Missed: {info.get('fraud_missed', 0)}")
            
            # Control speed for 4+ minute demo
            time.sleep(0.03)  # ~33 FPS, 8000 steps = ~4 minutes
        
        print("\n" + "=" * 60)
        print("Episode Complete!")
        print(f"Total Steps: {step}")
        print(f"Total Reward: {info.get('cumulative_reward', 0):.2f}")
        print(f"Fraud Caught: {info.get('fraud_caught', 0)}")
        print(f"Fraud Missed: {info.get('fraud_missed', 0)}")
        print(f"Correct Decisions: {info.get('correct_decisions', 0)}")
        print(f"Accuracy: {info.get('correct_decisions', 0) / max(1, step) * 100:.1f}%")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    finally:
        env.close()
        print("\nDemo closed. Thanks for watching! 🎮")


if __name__ == "__main__":
    main()

