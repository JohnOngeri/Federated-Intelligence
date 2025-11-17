# PrivFed Reinforcement Learning: Multi-Bank Fraud Detection

## 🎯 Mission
Simulate a multi-bank fraud detection system where an RL agent learns optimal transaction screening strategies under privacy budget and manual review constraints.

## 📁 Project Structure
\`\`\`
project_root/
├── environment/
│   ├── custom_env.py          # Custom Gymnasium environment
│   ├── rendering.py            # Pygame visualization
│   └── random_demo.py          # Random action demonstration
├── training/
│   ├── dqn_training.py         # DQN training with hyperparameter search
│   ├── pg_training.py          # PPO and A2C training
│   └── reinforce_training.py   # Manual REINFORCE implementation
├── models/
│   ├── dqn/                    # DQN model checkpoints
│   └── pg/                     # Policy gradient model checkpoints
├── results/
│   ├── training_logs/          # Training metrics
│   └── figures/                # Performance plots
├── main.py                     # Best model demonstration
├── report.md                   # Technical report
└── requirements.txt            # Dependencies
\`\`\`

## 🚀 Quick Start

### Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Run Random Demo (No Training)
\`\`\`bash
python environment/random_demo.py
\`\`\`

### Train All Algorithms
\`\`\`bash
# DQN
python training/dqn_training.py

# PPO and A2C
python training/pg_training.py

# REINFORCE
python training/reinforce_training.py
\`\`\`

### Run Best Model
\`\`\`bash
python main.py
\`\`\`

## 🎮 Environment Details

**Action Space:** Discrete(3)
- 0: APPROVE transaction
- 1: BLOCK transaction
- 2: MANUAL_REVIEW transaction

**Observation Space:** Box(13,) - Continuous features normalized [0,1]

**Reward Structure:**
- Legit + APPROVE: +1
- Legit + BLOCK: -5
- Fraud + APPROVE: -20
- Fraud + BLOCK: +10
- MANUAL_REVIEW: -1 (legit) or +5 (fraud)

## 📊 Results Summary

See `report.md` for full analysis and comparison of all algorithms.

## 🎓 Assignment Compliance

This project fulfills all requirements for the RL assignment including:
- ✅ Custom environment with exhaustive action space
- ✅ Rich observation space with domain-relevant features
- ✅ Pygame visualization
- ✅ 4 RL algorithms (DQN, PPO, A2C, REINFORCE)
- ✅ Hyperparameter tuning (10+ runs per algorithm)
- ✅ Performance metrics and analysis
- ✅ Technical report with graphs

## 📝 License
MIT License - Educational Project
