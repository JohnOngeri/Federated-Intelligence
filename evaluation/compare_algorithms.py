"""
Comprehensive algorithm comparison and visualization
Generates publication-quality plots and analysis tables
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def load_all_results():
    """Load results from all algorithms"""
    
    print("Loading training results...")
    
    results = {}
    
    # Check multiple locations for results
    result_paths = [
        ("results/training_logs/", "../results/training_logs/", "../scripts/results/training_logs/"),
    ]
    
    for base_paths in result_paths:
        for base_path in base_paths:
            # Load DQN
            dqn_path = os.path.join(base_path, "dqn_results.csv")
            if os.path.exists(dqn_path) and 'DQN' not in results:
                results['DQN'] = pd.read_csv(dqn_path)
                print(f"  ✓ DQN: {len(results['DQN'])} runs")
            
            # Load PPO
            ppo_path = os.path.join(base_path, "ppo_results.csv")
            if os.path.exists(ppo_path) and 'PPO' not in results:
                results['PPO'] = pd.read_csv(ppo_path)
                print(f"  ✓ PPO: {len(results['PPO'])} runs")
            
            # Load A2C
            a2c_path = os.path.join(base_path, "a2c_results.csv")
            if os.path.exists(a2c_path) and 'A2C' not in results:
                results['A2C'] = pd.read_csv(a2c_path)
                print(f"  ✓ A2C: {len(results['A2C'])} runs")
            
            # Load REINFORCE
            reinforce_path = os.path.join(base_path, "reinforce_results.csv")
            if os.path.exists(reinforce_path) and 'REINFORCE' not in results:
                results['REINFORCE'] = pd.read_csv(reinforce_path)
                print(f"  ✓ REINFORCE: {len(results['REINFORCE'])} runs")
    
    return results


def create_summary_table(results):
    """Create comprehensive summary table"""
    
    print("\nGenerating summary statistics...")
    
    summary_data = []
    
    for algo_name, df in results.items():
        summary_data.append({
            'Algorithm': algo_name,
            'Runs': len(df),
            'Best Reward': df['mean_reward'].max(),
            'Worst Reward': df['mean_reward'].min(),
            'Mean Reward': df['mean_reward'].mean(),
            'Std Reward': df['mean_reward'].std(),
            'Best Fraud Caught': df['mean_fraud_caught'].max() if 'mean_fraud_caught' in df else 0,
            'Mean Fraud Caught': df['mean_fraud_caught'].mean() if 'mean_fraud_caught' in df else 0,
            'Mean Fraud Missed': df['mean_fraud_missed'].mean() if 'mean_fraud_missed' in df else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best Reward', ascending=False)
    
    # Save to CSV
    os.makedirs("results/analysis", exist_ok=True)
    summary_df.to_csv("results/analysis/algorithm_summary.csv", index=False)
    
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))
    
    return summary_df


def plot_algorithm_comparison(results, summary_df):
    """Create comprehensive comparison figure"""
    
    print("\nGenerating comparison plots...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Box plot of reward distributions
    ax1 = fig.add_subplot(gs[0, :])
    data_for_box = []
    labels_for_box = []
    
    for algo_name, df in results.items():
        data_for_box.append(df['mean_reward'].values)
        labels_for_box.append(algo_name)
    
    bp = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax1.set_ylabel('Mean Evaluation Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Algorithm Performance Distribution (10 Runs Each)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Bar chart of best performance
    ax2 = fig.add_subplot(gs[1, 0])
    best_rewards = [df['mean_reward'].max() for df in results.values()]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(results)))
    
    bars = ax2.bar(results.keys(), best_rewards, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Best Mean Reward', fontsize=11, fontweight='bold')
    ax2.set_title('Peak Performance', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Fraud detection performance
    ax3 = fig.add_subplot(gs[1, 1])
    fraud_caught = [df['mean_fraud_caught'].mean() if 'mean_fraud_caught' in df else 0 
                    for df in results.values()]
    fraud_missed = [df['mean_fraud_missed'].mean() if 'mean_fraud_missed' in df else 0 
                    for df in results.values()]
    
    x = np.arange(len(results))
    width = 0.35
    
    ax3.bar(x - width/2, fraud_caught, width, label='Caught', color='green', alpha=0.7)
    ax3.bar(x + width/2, fraud_missed, width, label='Missed', color='red', alpha=0.7)
    ax3.set_ylabel('Average Fraud Count', fontsize=11, fontweight='bold')
    ax3.set_title('Fraud Detection Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(results.keys()))
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Stability (std dev)
    ax4 = fig.add_subplot(gs[1, 2])
    std_devs = [df['mean_reward'].std() for df in results.values()]
    
    bars = ax4.bar(results.keys(), std_devs, color='orange', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Std Dev of Rewards', fontsize=11, fontweight='bold')
    ax4.set_title('Training Stability', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Violin plot (reward distribution detail)
    ax5 = fig.add_subplot(gs[2, :2])
    
    all_data = []
    all_labels = []
    for algo_name, df in results.items():
        all_data.extend(df['mean_reward'].values)
        all_labels.extend([algo_name] * len(df))
    
    violin_df = pd.DataFrame({'Algorithm': all_labels, 'Reward': all_data})
    sns.violinplot(data=violin_df, x='Algorithm', y='Reward', ax=ax5, inner='box')
    ax5.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    ax5.set_title('Reward Distribution Detail (Violin Plot)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Summary metrics table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Algorithm'],
            f"{row['Best Reward']:.1f}",
            f"{row['Mean Reward']:.1f}",
            f"{row['Best Fraud Caught']:.1f}"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Algo', 'Best', 'Mean', 'Fraud↑'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('PrivFed Fraud Detection: Comprehensive Algorithm Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig('results/figures/comprehensive_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/figures/comprehensive_comparison.png")
    plt.close()


def plot_hyperparameter_sensitivity(results):
    """Analyze hyperparameter sensitivity"""
    
    print("\nGenerating hyperparameter analysis...")
    
    # For each algorithm, plot learning rate sensitivity
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (algo_name, df) in enumerate(results.items()):
        if idx >= 4:
            break
        
        ax = axes[idx]
        
        # Group by learning rate if available
        if 'learning_rate' in df.columns:
            grouped = df.groupby('learning_rate')['mean_reward'].agg(['mean', 'std'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', markersize=8, capsize=5, capthick=2,
                       linewidth=2, label=algo_name)
            
            ax.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
            ax.set_ylabel('Mean Reward', fontsize=11, fontweight='bold')
            ax.set_title(f'{algo_name}: Learning Rate Sensitivity', 
                        fontsize=12, fontweight='bold')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No LR data for {algo_name}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/hyperparameter_sensitivity.png', 
                dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/figures/hyperparameter_sensitivity.png")
    plt.close()


def generate_latex_table(summary_df):
    """Generate LaTeX table for academic reports"""
    
    print("\nGenerating LaTeX table...")
    
    latex_str = summary_df.to_latex(
        index=False,
        float_format="%.2f",
        column_format='l' + 'r' * (len(summary_df.columns) - 1),
        caption="Algorithm Performance Comparison",
        label="tab:algorithm_comparison"
    )
    
    with open("results/analysis/algorithm_table.tex", 'w') as f:
        f.write(latex_str)
    
    print("  ✓ Saved: results/analysis/algorithm_table.tex")


def main():
    """Main evaluation pipeline"""
    
    print("=" * 70)
    print("PrivFed: Comprehensive Algorithm Evaluation")
    print("=" * 70)
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("\nERROR: No training results found!")
        print("Please run training scripts first:")
        print("  python training/dqn_training.py")
        print("  python training/pg_training.py")
        print("  python training/reinforce_training.py")
        return
    
    # Generate summary table
    summary_df = create_summary_table(results)
    
    # Create visualizations
    plot_algorithm_comparison(results, summary_df)
    plot_hyperparameter_sensitivity(results)
    
    # Generate training curves and stability analysis
    try:
        print("\nGenerating training curves and stability analysis...")
        from evaluation.plot_training_curves import plot_comprehensive_training_analysis
        plot_comprehensive_training_analysis()
    except Exception as e:
        print(f"  Warning: Could not generate training curves: {e}")
        print("  Run separately: python evaluation/plot_training_curves.py")
    
    # Generate LaTeX table
    generate_latex_table(summary_df)
    
    # Identify best overall
    best_algo = summary_df.iloc[0]
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nBest Algorithm: {best_algo['Algorithm']}")
    print(f"  Best Reward: {best_algo['Best Reward']:.2f}")
    print(f"  Mean Reward: {best_algo['Mean Reward']:.2f} ± {best_algo['Std Reward']:.2f}")
    print(f"  Fraud Detection: {best_algo['Best Fraud Caught']:.1f} caught, "
          f"{best_algo['Mean Fraud Missed']:.1f} missed")
    
    print("\nAll analysis outputs:")
    print("  • results/analysis/algorithm_summary.csv")
    print("  • results/analysis/algorithm_table.tex")
    print("  • results/figures/comprehensive_comparison.png")
    print("  • results/figures/hyperparameter_sensitivity.png")
    print("  • results/figures/training_stability_comprehensive.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
