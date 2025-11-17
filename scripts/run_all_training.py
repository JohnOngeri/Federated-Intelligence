"""
Master script to run all training experiments sequentially
Useful for automated full experiment runs
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_path, name):
    """Run a training script and log output"""
    
    print("\n" + "=" * 70)
    print(f"Running: {name}")
    print(f"Script: {script_path}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"✓ {name} Complete")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("=" * 70)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR in {name}")
        print(f"Return code: {e.returncode}")
        return False

def main():
    """Run all training scripts"""
    
    print("=" * 70)
    print("PrivFed: Complete Training Pipeline")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will train:")
    print("  1. DQN (10 configurations)")
    print("  2. PPO (10 configurations)")
    print("  3. A2C (10 configurations)")
    print("  4. REINFORCE (10 configurations)")
    print("\nEstimated total time: 4-8 hours (depending on hardware)")
    print("=" * 70)
    
    input("\nPress ENTER to start training pipeline...")
    
    overall_start = time.time()
    
    # Training scripts
    scripts = [
        ("training/dqn_training.py", "DQN Training"),
        ("training/pg_training.py", "PPO & A2C Training"),
        ("training/reinforce_training.py", "REINFORCE Training"),
    ]
    
    results = []
    
    for script_path, name in scripts:
        if not os.path.exists(script_path):
            print(f"\n✗ ERROR: {script_path} not found!")
            continue
        
        success = run_script(script_path, name)
        results.append((name, success))
    
    # Run evaluation
    print("\n" + "=" * 70)
    print("Running Comprehensive Evaluation...")
    print("=" * 70)
    
    eval_success = run_script("evaluation/compare_algorithms.py", "Evaluation & Analysis")
    results.append(("Evaluation", eval_success))
    
    # Summary
    overall_time = time.time() - overall_start
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Total time: {overall_time/3600:.2f} hours")
    print("\nResults:")
    
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    all_success = all(s for _, s in results)
    
    if all_success:
        print("\n🎉 All training and evaluation completed successfully!")
        print("\nNext steps:")
        print("  1. Review results in results/analysis/")
        print("  2. Check visualizations in results/figures/")
        print("  3. Run best model: python main.py")
    else:
        print("\n⚠ Some training runs failed. Check logs above.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
