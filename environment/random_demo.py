"""
random_demo.py

High-quality random-policy demonstration for the Federated Learning Fraud Environment.
This script:
 - Runs N episodes using pure random actions (no learning)
 - Renders the environment (optional)
 - Logs episode statistics to console AND CSV
 - Handles clean quitting when pygame window is closed
 - Provides a command-line interface for reproducibility

This acts as the "baseline" agent for your RL assignment.
"""

import argparse
import csv
import os
import time
from typing import Dict, Any, List

from custom_env import PrivFedFraudEnv  # Update to your new project name if needed


# ------------------------------------------------------------
# Utility: Safe extraction of metrics from info dict
# ------------------------------------------------------------
def safe_get(info: Dict[str, Any], key: str, default: Any = 0) -> Any:
    return info[key] if key in info and info[key] is not None else default


# ------------------------------------------------------------
# Run a single episode with random actions
# ------------------------------------------------------------
def run_random_episode(
    env: PrivFedFraudEnv,
    episode_index: int,
    render: bool,
    sleep: float
) -> Dict[str, Any]:
    """
    Runs one complete episode using random actions.

    Returns:
        dict -> episode summary statistics
    """
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    step_count = 0
    action_counts = {"APPROVE": 0, "BLOCK": 0, "MANUAL_REVIEW": 0}

    action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL_REVIEW"}

    print(f"\n================= Episode {episode_index + 1} =================")
    print(f"{'Step':<6} {'Action':<15} {'Reward':<10} {'Total':<10}")
    print("-" * 52)

    while not done:

        # Render if enabled
        if render:
            quit_signal = env.render()
            if quit_signal:   # environment signals quit
                print("\n[INFO] Window closed. Exiting episode early.")
                done = True
                break

        # Random action
        action = env.action_space.sample()
        action_name = action_names.get(action, f"ACTION_{action}")
        action_counts[action_name] += 1

        # Store for rendering if available
        if hasattr(env, "renderer") and env.renderer:
            if hasattr(env.renderer, "set_action"):
                env.renderer.set_action(action)

        # Apply action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        step_count += 1
        total_reward += reward

        print(f"{step_count:<6} {action_name:<15} {reward:<10.2f} {total_reward:<10.2f}")

        if sleep > 0:
            time.sleep(sleep)

    # Safe metrics retrieval
    fraud_caught = safe_get(info, "fraud_caught")
    fraud_missed = safe_get(info, "fraud_missed")
    correct = safe_get(info, "correct_decisions")

    print("-" * 52)
    print("Episode Complete!")
    print(f"Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Correct Decisions: {correct}")
    print(f"Fraud Caught: {fraud_caught}")
    print(f"Fraud Missed: {fraud_missed}")

    return {
        "episode": episode_index + 1,
        "steps": step_count,
        "total_reward": total_reward,
        "correct_decisions": correct,
        "fraud_caught": fraud_caught,
        "fraud_missed": fraud_missed,
        "approve_count": action_counts["APPROVE"],
        "block_count": action_counts["BLOCK"],
        "review_count": action_counts["MANUAL_REVIEW"],
    }


# ------------------------------------------------------------
# Save CSV summary
# ------------------------------------------------------------
def save_csv(rows: List[Dict[str, Any]], filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fieldnames = [
        "episode", "steps", "total_reward",
        "correct_decisions", "fraud_caught", "fraud_missed",
        "approve_count", "block_count", "review_count"
    ]

    write_header = not os.path.exists(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"\n[CSV] Saved random demo results to: {filepath}\n")


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Random-policy baseline demo for the Federated Fraud Environment")
    parser.add_argument("--episodes", type=int, default=1, help="How many random episodes to run")
    parser.add_argument("--sleep", type=float, default=0.3, help="Delay between steps for visualization")
    parser.add_argument("--no-render", action="store_true", help="Disable pygame rendering")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")

    args = parser.parse_args()

    render = not args.no_render

    print("=" * 60)
    print("Federated Learning – Random Policy Demonstration")
    print("=" * 60)
    print("\nThis script demonstrates the environment using RANDOM actions only.")
    print("Used for visualization and baseline comparison.\n")

    # Create environment
    env = PrivFedFraudEnv(render_mode="human" if render else None, max_steps=args.max_steps)

    # Seed if provided
    if args.seed is not None:
        env.reset(seed=args.seed)

    all_episode_summaries = []

    for ep in range(args.episodes):
        summary = run_random_episode(env, ep, render, args.sleep)
        all_episode_summaries.append(summary)

    env.close()

    # Save results
    save_csv(all_episode_summaries, "results/random_demo.csv")

    # Print overall summary
    avg_reward = sum(ep["total_reward"] for ep in all_episode_summaries) / len(all_episode_summaries)
    print("=" * 60)
    print("Overall Summary Across Episodes")
    print("=" * 60)
    print(f"Episodes run: {args.episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print("CSV log saved in results/random_demo.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
