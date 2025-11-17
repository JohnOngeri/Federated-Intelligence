"""
Pygame visualization for PrivFedFraudEnv
Displays banks, transactions, actions, and environment state
"""

import pygame
import numpy as np
from typing import Optional


class PrivFedRenderer:
    """Pygame-based visualization for the fraud detection environment"""
    
    def __init__(self, env):
        self.env = env
        
        # Window settings
        self.window_width = 1200
        self.window_height = 800
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        
        # Colors
        self.COLOR_BG = (245, 245, 250)
        self.COLOR_BANK = (70, 130, 180)
        self.COLOR_BANK_ACTIVE = (255, 140, 0)
        self.COLOR_FRAUD = (220, 20, 60)
        self.COLOR_LEGIT = (50, 205, 50)
        self.COLOR_APPROVE = (100, 200, 100)
        self.COLOR_BLOCK = (200, 100, 100)
        self.COLOR_REVIEW = (255, 200, 50)
        self.COLOR_TEXT = (30, 30, 30)
        self.COLOR_PANEL = (255, 255, 255)
        
        # Layout
        self.bank_positions = []
        self._calculate_layout()
        
        # Animation
        self.last_action = None
        self.action_display_timer = 0
        
        self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame components"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("PrivFed Fraud Detection - Multi-Bank Simulation")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 20)
    
    def _calculate_layout(self):
        """Calculate positions for banks and UI elements"""
        
        # Bank positions in a circle
        num_banks = self.env.num_banks
        center_x = 300
        center_y = 400
        radius = 180
        
        for i in range(num_banks):
            angle = 2 * np.pi * i / num_banks - np.pi / 2
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            self.bank_positions.append((int(x), int(y)))
    
    def render(self):
        """Main render function"""
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
        
        # Clear screen
        self.screen.fill(self.COLOR_BG)
        
        # Draw components
        self._draw_banks()
        self._draw_transaction_panel()
        self._draw_action_indicator()
        self._draw_stats_panel()
        self._draw_budgets()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.env.metadata['render_fps'])
        
        # Decrement action timer
        if self.action_display_timer > 0:
            self.action_display_timer -= 1
        
        return None
    
    def _draw_banks(self):
        """Draw the 5 banks"""
        
        for i, (x, y) in enumerate(self.bank_positions):
            # Check if this is the current bank
            is_active = (self.env.current_transaction and 
                        self.env.current_transaction['bank_id'] == i)
            
            color = self.COLOR_BANK_ACTIVE if is_active else self.COLOR_BANK
            
            # Draw bank circle
            pygame.draw.circle(self.screen, color, (x, y), 50)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, (x, y), 50, 3)
            
            # Draw bank label
            label = self.font_small.render(f"Bank {i+1}", True, self.COLOR_TEXT)
            label_rect = label.get_rect(center=(x, y - 70))
            self.screen.blit(label, label_rect)
            
            # Draw fraud rate
            fraud_rate = self.env.bank_fraud_rates[i]
            rate_text = self.font_tiny.render(f"Fraud: {fraud_rate:.1%}", True, self.COLOR_TEXT)
            rate_rect = rate_text.get_rect(center=(x, y + 70))
            self.screen.blit(rate_text, rate_rect)
    
    def _draw_transaction_panel(self):
        """Draw current transaction details"""
        
        panel_x = 650
        panel_y = 50
        panel_width = 500
        panel_height = 250
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.COLOR_PANEL, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("Current Transaction", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        if self.env.current_transaction is None:
            return
        
        txn = self.env.current_transaction
        y_offset = panel_y + 70
        line_height = 35
        
        # Transaction details
        details = [
            f"Bank: Bank {txn['bank_id'] + 1}",
            f"Amount: ${txn['amount']:.2f}",
            f"Type: {self.env.transaction_types[txn['txn_type']].title()}",
            f"Risk Score: {txn['risk_score']:.2f}",
        ]
        
        for detail in details:
            text = self.font_small.render(detail, True, self.COLOR_TEXT)
            self.screen.blit(text, (panel_x + 30, y_offset))
            y_offset += line_height
        
        # Ground truth (visible to human, not agent)
        truth_color = self.COLOR_FRAUD if txn['is_fraud'] else self.COLOR_LEGIT
        truth_text = "🚨 FRAUD" if txn['is_fraud'] else "✓ LEGITIMATE"
        truth_surface = self.font.render(truth_text, True, truth_color)
        self.screen.blit(truth_surface, (panel_x + 250, panel_y + 180))
    
    def _draw_action_indicator(self):
        """Draw the agent's chosen action"""
        
        panel_x = 650
        panel_y = 330
        panel_width = 500
        panel_height = 120
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.COLOR_PANEL, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("Agent Action", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        # Display last action
        if self.last_action is not None:
            action_names = {
                0: "APPROVE",
                1: "BLOCK",
                2: "MANUAL REVIEW"
            }
            action_colors = {
                0: self.COLOR_APPROVE,
                1: self.COLOR_BLOCK,
                2: self.COLOR_REVIEW
            }
            
            action_text = action_names.get(self.last_action, "UNKNOWN")
            action_color = action_colors.get(self.last_action, self.COLOR_TEXT)
            
            action_surface = self.font.render(action_text, True, action_color)
            action_rect = action_surface.get_rect(center=(panel_x + 250, panel_y + 75))
            self.screen.blit(action_surface, action_rect)
    
    def _draw_stats_panel(self):
        """Draw performance statistics"""
        
        panel_x = 650
        panel_y = 480
        panel_width = 500
        panel_height = 280
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.COLOR_PANEL, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("Episode Statistics", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        y_offset = panel_y + 70
        line_height = 35
        
        # Statistics
        stats = [
            f"Step: {self.env.current_step} / {self.env.max_steps}",
            f"Cumulative Reward: {self.env.cumulative_reward:.1f}",
            f"Correct Decisions: {self.env.correct_decisions}",
            f"Fraud Caught: {self.env.total_fraud_caught}",
            f"Fraud Missed: {self.env.total_fraud_missed}",
        ]
        
        for stat in stats:
            text = self.font_small.render(stat, True, self.COLOR_TEXT)
            self.screen.blit(text, (panel_x + 30, y_offset))
            y_offset += line_height
    
    def _draw_budgets(self):
        """Draw privacy and manual review budgets"""
        
        panel_x = 50
        panel_y = 650
        panel_width = 500
        panel_height = 120
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.COLOR_PANEL, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title
        title = self.font.render("Budgets", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 20, panel_y + 15))
        
        # Privacy budget bar
        privacy_used_ratio = min(self.env.privacy_budget_used / self.env.initial_privacy_budget, 1.0)
        bar_width = 400
        bar_height = 25
        bar_x = panel_x + 90
        bar_y = panel_y + 60
        
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (255, 100, 100), 
                        (bar_x, bar_y, int(bar_width * privacy_used_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        privacy_label = self.font_small.render("Privacy:", True, self.COLOR_TEXT)
        self.screen.blit(privacy_label, (panel_x + 10, bar_y))
        
        # Manual review budget
        manual_ratio = self.env.manual_budget_remaining / self.env.initial_manual_budget
        bar_y += 40
        
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (100, 150, 255), 
                        (bar_x, bar_y, int(bar_width * manual_ratio), bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, 
                        (bar_x, bar_y, bar_width, bar_height), 2)
        
        manual_label = self.font_small.render("Manual:", True, self.COLOR_TEXT)
        self.screen.blit(manual_label, (panel_x + 10, bar_y))
        
        manual_count = self.font_tiny.render(
            f"{self.env.manual_budget_remaining}/{self.env.initial_manual_budget}", 
            True, self.COLOR_TEXT
        )
        self.screen.blit(manual_count, (bar_x + bar_width + 10, bar_y + 3))
    
    def set_action(self, action):
        """Set the action to display"""
        self.last_action = action
        self.action_display_timer = 10
    
    def close(self):
        """Close pygame window"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
