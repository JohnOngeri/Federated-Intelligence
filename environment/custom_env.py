"""
PrivFedFraudEnv: Custom Gymnasium Environment
Simulates multi-bank fraud detection with privacy and review budget constraints
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    # When imported from project root
    from environment.rendering import PrivFedRenderer
except ModuleNotFoundError:
    try:
        # When running scripts from within the environment package
        from rendering import PrivFedRenderer
    except ModuleNotFoundError:
        PrivFedRenderer = None


class PrivFedFraudEnv(gym.Env):
    """
    Multi-Bank Fraud Detection Environment
    
    The agent must decide whether to APPROVE, BLOCK, or send to MANUAL_REVIEW
    for each transaction, balancing fraud detection with privacy and cost constraints.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    # Action definitions
    ACTION_APPROVE = 0
    ACTION_BLOCK = 1
    ACTION_MANUAL_REVIEW = 2
    
    def __init__(self, render_mode=None, max_steps=100):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Action space: APPROVE, BLOCK, MANUAL_REVIEW
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 13 features (all normalized 0-1)
        # [bank_id_onehot(5), amount, txn_type, risk_score, bank_fraud_rate, 
        #  privacy_budget_used, manual_budget_remaining, step_normalized]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )
        
        # Environment parameters
        self.num_banks = 5
        self.initial_privacy_budget = 100.0
        self.initial_manual_budget = 20
        self.privacy_cost_per_action = 1.0
        
        # Bank characteristics (Non-IID: different fraud rates)
        self.bank_fraud_rates = np.array([0.05, 0.15, 0.08, 0.20, 0.10])
        
        # Transaction types
        self.transaction_types = ['transfer', 'withdrawal', 'purchase', 'payment']
        
        # State variables
        self.current_step = 0
        self.privacy_budget_used = 0.0
        self.manual_budget_remaining = 0
        self.cumulative_reward = 0.0
        
        # Current transaction
        self.current_transaction = None
        self.current_bank_id = 0
        
        # Statistics
        self.correct_decisions = 0
        self.total_fraud_caught = 0
        self.total_fraud_missed = 0
        
        # Rendering
        self.renderer = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset counters
        self.current_step = 0
        self.privacy_budget_used = 0.0
        self.manual_budget_remaining = self.initial_manual_budget
        self.cumulative_reward = 0.0
        
        # Reset statistics
        self.correct_decisions = 0
        self.total_fraud_caught = 0
        self.total_fraud_missed = 0
        
        # Generate first transaction
        self._generate_transaction()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        
        # Validate action
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Get reward for this action
        reward = self._calculate_reward(action)
        self.cumulative_reward += reward
        
        # Update budgets
        self.privacy_budget_used += self.privacy_cost_per_action
        if action == self.ACTION_MANUAL_REVIEW:
            self.manual_budget_remaining -= 1
        
        # Track statistics
        self._update_statistics(action)
        
        # Increment step
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Generate next transaction (if not done)
        if not (terminated or truncated):
            self._generate_transaction()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_transaction(self):
        """Generate a new transaction for evaluation"""
        
        # Select bank (Non-IID: some banks have more transactions)
        bank_weights = np.array([0.15, 0.25, 0.20, 0.30, 0.10])
        self.current_bank_id = self.np_random.choice(self.num_banks, p=bank_weights)
        
        # Determine if fraud based on bank's fraud rate
        is_fraud = self.np_random.random() < self.bank_fraud_rates[self.current_bank_id]
        
        # Generate transaction features
        if is_fraud:
            # Fraudulent transactions: higher amounts, higher risk scores
            amount = self.np_random.uniform(500, 5000)
            risk_score = self.np_random.uniform(0.6, 1.0)
        else:
            # Legitimate transactions: lower amounts, lower risk scores
            amount = self.np_random.uniform(10, 1000)
            risk_score = self.np_random.uniform(0.0, 0.5)
        
        txn_type = self.np_random.choice(len(self.transaction_types))
        
        self.current_transaction = {
            'bank_id': self.current_bank_id,
            'amount': amount,
            'txn_type': txn_type,
            'risk_score': risk_score,
            'is_fraud': is_fraud
        }
    
    def _get_observation(self):
        """Construct observation vector"""
        
        if self.current_transaction is None:
            return np.zeros(13, dtype=np.float32)
        
        txn = self.current_transaction
        
        # One-hot encode bank ID
        bank_onehot = np.zeros(self.num_banks, dtype=np.float32)
        bank_onehot[txn['bank_id']] = 1.0
        
        # Normalize features
        amount_norm = min(txn['amount'] / 5000.0, 1.0)
        txn_type_norm = txn['txn_type'] / len(self.transaction_types)
        risk_score_norm = txn['risk_score']
        bank_fraud_rate_norm = self.bank_fraud_rates[txn['bank_id']]
        privacy_budget_norm = min(self.privacy_budget_used / self.initial_privacy_budget, 1.0)
        manual_budget_norm = self.manual_budget_remaining / self.initial_manual_budget
        step_norm = self.current_step / self.max_steps
        
        observation = np.concatenate([
            bank_onehot,                      # 5 features
            [amount_norm],                    # 1 feature
            [txn_type_norm],                  # 1 feature
            [risk_score_norm],                # 1 feature
            [bank_fraud_rate_norm],           # 1 feature
            [privacy_budget_norm],            # 1 feature
            [manual_budget_norm],             # 1 feature
            [step_norm]                       # 1 feature
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self, action):
        """Calculate reward based on action and ground truth"""
        
        is_fraud = self.current_transaction['is_fraud']
        
        # Reward matrix
        if action == self.ACTION_APPROVE:
            if is_fraud:
                reward = -20.0  # Bad: approved fraud
            else:
                reward = 1.0    # Good: approved legit
        
        elif action == self.ACTION_BLOCK:
            if is_fraud:
                reward = 10.0   # Good: blocked fraud
            else:
                reward = -5.0   # Bad: blocked legit (customer friction)
        
        elif action == self.ACTION_MANUAL_REVIEW:
            if is_fraud:
                reward = 5.0    # Decent: caught fraud but used resources
            else:
                reward = -1.0   # Minor cost: wasted manual review
        
        # Add penalties for resource usage
        privacy_penalty = -0.05 * (self.privacy_budget_used / self.initial_privacy_budget)
        
        return reward + privacy_penalty
    
    def _update_statistics(self, action):
        """Track performance statistics"""
        
        is_fraud = self.current_transaction['is_fraud']
        
        # Count correct decisions
        if is_fraud and action in [self.ACTION_BLOCK, self.ACTION_MANUAL_REVIEW]:
            self.correct_decisions += 1
            self.total_fraud_caught += 1
        elif not is_fraud and action == self.ACTION_APPROVE:
            self.correct_decisions += 1
        
        # Count missed fraud
        if is_fraud and action == self.ACTION_APPROVE:
            self.total_fraud_missed += 1
    
    def _check_termination(self):
        """Check if episode should terminate early"""
        
        # Terminate if budgets exhausted
        if self.manual_budget_remaining <= 0:
            return True
        
        if self.privacy_budget_used >= self.initial_privacy_budget:
            return True
        
        return False
    
    def _get_info(self):
        """Return additional information"""
        
        info = {
            'step': self.current_step,
            'privacy_budget_used': self.privacy_budget_used,
            'manual_budget_remaining': self.manual_budget_remaining,
            'cumulative_reward': self.cumulative_reward,
            'correct_decisions': self.correct_decisions,
            'fraud_caught': self.total_fraud_caught,
            'fraud_missed': self.total_fraud_missed
        }
        
        if self.current_transaction:
            info['current_bank'] = self.current_transaction['bank_id']
            info['is_fraud'] = self.current_transaction['is_fraud']
        
        return info
    
    def render(self):
        """Render the environment"""
        
        if self.render_mode == 'human':
            if PrivFedRenderer is None:
                raise ModuleNotFoundError(
                    "PrivFedRenderer is unavailable. Ensure pygame and the "
                    "rendering module dependencies are installed."
                )
            
            if self.renderer is None:
                self.renderer = PrivFedRenderer(self)
            
            return self.renderer.render()
        
        return None
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()
