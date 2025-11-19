"""
Master script to run all training experiments sequentially.
Adds:
  ✓ Resume training if interrupted
  ✓ CSV logging of progress
  ✓ Safe path handling for Colab and local
"""

import os
import sys
import csv
import subprocess
import time
from datetime import datetime

# Base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# CSV progress log file
PROGRESS_FILE = os.path.join(BASE_DIR, "results", "training_progress.csv")


def load_progress():
    """Load progress CSV to know which scripts already ran."""
    if not os.path.exists(PROGRESS_FILE):
        return {}

    progress = {}
    with open(PROGRESS_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            progress[row["script_name"]] = row["status"]

    return progress


def save_progress(script_name, status):
    """Append a row to the progress CSV."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

    file_exists = os.path.exists(PROGRESS_FILE)

    with open(PROGRESS_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header only once
        if not file_exists:
            writer.writerow(["script_name", "status", "timestamp"])

        writer.writerow([script_name, status, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def run_script(relative_path, script_name):
    """Run a script and stream output live."""
    script_path = os.path.join(BASE_DIR, relative_path)

    print("\n" + "=" * 70)
    print(f"Running: {script_name}")
    print(f"Script: {script_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    if not os.path.exists(script_path):
        print(f"✗ ERROR: Script not found -> {script_path}")
        save_progress(script_name, "NOT_FOUND")
        return False

    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, script_path],
            check=True
        )

        elapsed = (time.time() - start_time) / 60
        print("\n" + "=" * 70)
        print(f"✓ {script_name} Completed Successfully")
        print(f"Time Taken: {elapsed:.2f} minutes")
        print("=" * 70)

        save_progress(script_name, "SUCCESS")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR while running {script_name}")
        print(f"Return code: {e.returncode}")
        print("=" * 70)

        save_progress(script_name, "FAILED")
        return False


def main():
    """Run DQN → PPO/A2C → REINFORCE → Evaluation with resume support."""

    print("=" * 70)
    print("Federated Intelligence: COMPLETE TRAINING PIPELINE (with Resume)")
    print("=" * 70)

    progress = load_progress()
    print(f"Loaded progress log with {len(progress)} entries.")

    input("\nPress ENTER to begin or resume training... ")

    ALL_SCRIPTS = [
        ("training/dqn_training.py",        "DQN Training"),
        ("training/pg_training.py",         "PPO + A2C Training"),
        ("training/reinforce_training.py",  "REINFORCE Training"),
        ("evaluation/compare_algorithms.py","Evaluation & Analysis")
    ]

    for path, name in ALL_SCRIPTS:

        # Resume logic: skip scripts already marked SUCCESS
        if progress.get(name) == "SUCCESS":
            print(f"➡ Skipping {name} (already completed)")
            continue

        print(f"➡ Starting {name}...")
        success = run_script(path, name)

        if not success:
            print("\n⚠ Training stopped due to error.")
            print("Next run will RESUME from this exact point.")
            return

    print("\n🎉 All scripts completed successfully!")
    print(f"Training progress saved to: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
