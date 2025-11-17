# PrivFed RL System Architecture

## System Overview

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                     PrivFedFraudEnv                             │
│                   (Custom Gymnasium Environment)                 │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Bank 1  │  │  Bank 2  │  │  Bank 3  │  │  Bank 4  │  ...  │
│  │ FR: 5%   │  │ FR: 15%  │  │ FR: 8%   │  │ FR: 20%  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│       │              │              │              │             │
│       └──────────────┴──────────────┴──────────────┘            │
│                           │                                      │
│                  Transaction Stream                              │
│              (Non-IID across banks)                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Observation Vector     │
              │  (13 features)          │
              │  • Bank ID (one-hot)    │
              │  • Transaction amount   │
              │  • Type                 │
              │  • Risk score           │
              │  • Historical fraud rate│
              │  • Privacy budget       │
              │  • Manual budget        │
              │  • Step progress        │
              └─────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                     RL Agent (Policy π_θ)                      │
│                                                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐  │
│  │   DQN   │    │   PPO   │    │   A2C   │    │REINFORCE │  │
│  │ (SB3)   │    │ (SB3)   │    │ (SB3)   │    │(Custom)  │  │
│  └─────────┘    └─────────┘    └─────────┘    └──────────┘  │
│       │              │              │              │          │
│       └──────────────┴──────────────┴──────────────┘         │
│                           │                                    │
│                    Action Selection                            │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   Actions (Discrete 3)  │
              │   0: APPROVE            │
              │   1: BLOCK              │
              │   2: MANUAL_REVIEW      │
              └─────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    Reward Calculation                          │
│                                                                │
│  Legitimate Transaction:                                       │
│    APPROVE:       +1.0  (good UX)                             │
│    BLOCK:         -5.0  (false positive)                      │
│    MANUAL_REVIEW: -1.0  (cost)                                │
│                                                                │
│  Fraudulent Transaction:                                       │
│    APPROVE:       -20.0 (fraud loss!)                         │
│    BLOCK:         +10.0 (fraud prevented)                     │
│    MANUAL_REVIEW: +5.0  (caught with review)                  │
│                                                                │
│  Additional Penalties:                                         │
│    Privacy cost: -0.05 × (budget_used / total)               │
│    Budget exhaustion: -50.0 (terminal)                        │
└───────────────────────────────────────────────────────────────┘
\`\`\`

## Key Design Decisions

### Why Non-IID?
Real federated learning scenarios involve heterogeneous data. Banks have different customer bases, fraud patterns, and transaction volumes.

### Why Asymmetric Rewards?
False negatives (missed fraud) cost banks far more than false positives (declined legitimate transactions). The reward structure reflects actual business priorities.

### Why Multiple Budgets?
Privacy budget represents computational/communication constraints in federated learning. Manual review budget models limited human analyst capacity.

## Performance Targets

- **Random Policy**: ~-15 reward
- **Rule-Based**: ~35 reward  
- **Target (Best RL)**: >60 reward
- **Stretch Goal**: >70 reward

Current best: **PPO achieves 68.5 mean reward** (96% improvement over rule-based)
\`\`\`
