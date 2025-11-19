"""
High-end Pygame visualization for PrivFedFraudEnv.

This renderer is designed for a competitive, portfolio-grade RL project:
- Gradient background with depth
- Card-style UI with soft shadows
- Circular multi-bank layout with fraud "heat" halos and bevel
- Animated highlighting of the active bank
- Zoom/pulse effect on the last action (feels like "zoom-in" feedback)
- Step progress bar (episode timeline)
- Budget bars for privacy and manual review with refined colors
- Side panel with transaction + stats cards and icons
- Optional debug overlay (F3)
- Side panel toggle (TAB)
- Clean quit handling (ESC or window close)
- Resolution-aware layout (resizes with the window)

Usage in your environment:

    from .rendering import PrivFedRenderer

    class PrivFedFraudEnv(...):
        def __init__(...):
            ...
            self.renderer = None
            self.metadata = {"render_fps": 30}

        def render(self):
            if self.renderer is None:
                self.renderer = PrivFedRenderer(self)
            return self.renderer.render()

Optional (for outcome flash):
    # in your env.step() after computing outcome
    self.last_outcome = "fraud_caught"  # or "fraud_missed" or "correct_legit"
    self.outcome_pulse = 1.0
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import pygame

Color = Tuple[int, int, int]


class PrivFedRenderer:
    """Pygame-based visualization for a multi-bank federated fraud environment."""

    def __init__(
        self,
        env: Any,
        window_size: Tuple[int, int] = (1280, 800),
    ) -> None:
        """
        Initialize the renderer.

        Args:
            env: The environment instance (must expose certain attributes, see below).
            window_size: Initial window width/height.
        """
        self.env = env

        # Window & layout
        self.window_width, self.window_height = window_size
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        # Fonts
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_tiny: Optional[pygame.font.Font] = None

        # UI flags
        self.show_side_panel: bool = True
        self.show_debug: bool = False
        self._should_quit: bool = False

        # Colors – soft dashboard palette
        self.COLOR_BG_TOP: Color = (233, 240, 255)
        self.COLOR_BG_BOTTOM: Color = (245, 246, 250)

        self.COLOR_PANEL: Color = (255, 255, 255)
        self.COLOR_PANEL_BORDER: Color = (215, 215, 226)
        self.COLOR_TEXT: Color = (35, 35, 50)

        self.COLOR_BANK: Color = (70, 130, 180)
        self.COLOR_BANK_ACTIVE: Color = (255, 160, 60)
        self.COLOR_BANK_OUTLINE: Color = (40, 40, 55)

        self.COLOR_FRAUD: Color = (220, 30, 80)
        self.COLOR_LEGIT: Color = (40, 200, 100)

        self.COLOR_APPROVE: Color = (90, 190, 110)
        self.COLOR_BLOCK: Color = (215, 90, 90)
        self.COLOR_REVIEW: Color = (255, 205, 85)

        # Budget bar colors
        self.COLOR_PRIVACY_BAR: Color = (255, 107, 107)
        self.COLOR_MANUAL_BAR: Color = (93, 173, 236)

        # Layout elements
        self.bank_positions: list[Tuple[int, int]] = []

        # Animation state
        self.last_action: Optional[int] = None
        self.action_pulse: float = 0.0
        self.action_pulse_decay: float = 0.90
        self.active_bank_pulse_phase: float = 0.0
        self.active_bank_pulse_speed: float = 0.11

        # Initialize pygame + compute layout
        self._init_pygame()
        self._calculate_layout()

    # ------------------------------------------------------------------
    # Initialization & layout
    # ------------------------------------------------------------------
    def _init_pygame(self) -> None:
        """Initialize pygame window, clock and fonts."""
        pygame.init()
        pygame.display.set_caption("Federated Fraud Detection - Multi-Bank Simulation")

        # RESIZABLE window for scaling
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE,
        )
        self.clock = pygame.time.Clock()

        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)

    def _calculate_layout(self) -> None:
        """Calculate positions for banks and panels based on window size."""
        if self.screen is None:
            return

        self.window_width, self.window_height = self.screen.get_size()

        # Banks arranged in a circle on the left half
        num_banks = getattr(self.env, "num_banks", 5)
        center_x = int(self.window_width * 0.28)
        center_y = int(self.window_height * 0.46)
        radius = int(min(self.window_width, self.window_height) * 0.20)

        self.bank_positions.clear()
        for i in range(num_banks):
            angle = 2 * math.pi * i / max(num_banks, 1) - math.pi / 2
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            self.bank_positions.append((x, y))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def render(self) -> bool:
        """
        Render one frame.

        Returns:
            bool: True if the user requested quit (window closed or ESC),
                  False otherwise. Caller should handle shutdown if True.
        """
        if self.screen is None or self.clock is None:
            return False

        self._handle_events()
        if self._should_quit:
            return True

        # Background & global elements
        self._draw_background_gradient()
        self._draw_step_progress()

        # Left side: banks + budgets
        self._draw_banks()
        self._draw_budgets()

        # Right side: panels, if enabled
        if self.show_side_panel:
            self._draw_transaction_panel()
            self._draw_action_panel()
            self._draw_stats_panel()

        # Mode tag and optional debug overlay
        self._draw_mode_tag()
        if self.show_debug:
            self._draw_debug_overlay()

        # Outcome flash overlay (optional; depends on env attrs)
        self._draw_outcome_flash()

        # Animation update
        self._update_animation_state()

        # Display update
        target_fps = getattr(self.env, "metadata", {}).get("render_fps", 30)
        pygame.display.flip()
        self.clock.tick(target_fps)

        return False

    def set_action(self, action: int) -> None:
        """Set the last chosen action to display and trigger a pulse animation."""
        self.last_action = action
        self.action_pulse = 1.0

    def close(self) -> None:
        """Close the pygame window gracefully."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _handle_events(self) -> None:
        """Process pygame events (resize, quit, keybinds)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True

            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.screen = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )
                self._calculate_layout()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_quit = True
                elif event.key == pygame.K_TAB:
                    self.show_side_panel = not self.show_side_panel
                elif event.key == pygame.K_F3:
                    self.show_debug = not self.show_debug

    # ------------------------------------------------------------------
    # Generic drawing helpers
    # ------------------------------------------------------------------
    def _draw_background_gradient(self) -> None:
        """Draw a vertical gradient background."""
        assert self.screen is not None
        w, h = self.screen.get_size()
        r1, g1, b1 = self.COLOR_BG_TOP
        r2, g2, b2 = self.COLOR_BG_BOTTOM

        for y in range(h):
            t = y / max(h - 1, 1)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (w, y))

    def _draw_card(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        radius: int = 14,
        shadow: bool = True,
        fill: Optional[Color] = None,
    ) -> pygame.Rect:
        """
        Draw a panel-like card with optional soft shadow and rounded corners.

        Returns:
            The pygame.Rect for the card (content area).
        """
        assert self.screen is not None
        fill = fill or self.COLOR_PANEL

        # Shadow (soft, subtle)
        if shadow:
            shadow_offset = 4
            shadow_rect = pygame.Rect(x + shadow_offset, y + shadow_offset, width, height)
            shadow_surf = pygame.Surface((width, height), pygame.SRCALPHA)
            pygame.draw.rect(
                shadow_surf,
                (0, 0, 0, 40),
                shadow_surf.get_rect(),
                border_radius=radius + 2,
            )
            self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Main card
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(
            self.screen,
            self.COLOR_PANEL_BORDER,
            rect,
            1,
            border_radius=radius,
        )
        return rect

    def _draw_debug_overlay(self) -> None:
        """Draw FPS and basic debug info in the top-left corner."""
        assert self.screen is not None and self.clock is not None and self.font_tiny is not None

        fps = self.clock.get_fps()
        text = f"FPS: {fps:5.1f}"
        surface = self.font_tiny.render(text, True, (40, 40, 60))
        self.screen.blit(surface, (10, 10))

    def _draw_step_progress(self) -> None:
        """Draw episode step progress bar like a subtle timeline at the top."""
        assert self.screen is not None and self.font_tiny is not None

        current_step = getattr(self.env, "current_step", 0)
        max_steps = max(1, getattr(self.env, "max_steps", 1))

        margin = 22
        bar_width = self.window_width - 2 * margin
        bar_height = 8
        bar_x = margin
        bar_y = 12

        # Background bar
        pygame.draw.rect(
            self.screen,
            (220, 225, 235),
            (bar_x, bar_y, bar_width, bar_height),
            border_radius=6,
        )

        ratio = current_step / max_steps
        pygame.draw.rect(
            self.screen,
            (120, 160, 255),
            (bar_x, bar_y, int(bar_width * ratio), bar_height),
            border_radius=6,
        )

        # Glossy highlight
        highlight_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height // 2)
        highlight = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
        highlight.fill((255, 255, 255, 40))
        self.screen.blit(highlight, highlight_rect.topleft)

        text = self.font_tiny.render(
            f"Step {current_step}/{max_steps}", True, (70, 70, 100)
        )
        self.screen.blit(text, (bar_x, bar_y + bar_height + 4))

    def _draw_mode_tag(self) -> None:
        """Draw a small mode tag in the top-right corner (e.g., RANDOM POLICY)."""
        assert self.screen is not None and self.font_tiny is not None

        mode_label = getattr(self.env, "mode_label", "RANDOM POLICY")
        padding_x = 10
        padding_y = 4

        surf = self.font_tiny.render(mode_label, True, (255, 255, 255))
        rect = surf.get_rect()
        rect.x = self.window_width - rect.width - padding_x * 3
        rect.y = 14

        bg_rect = pygame.Rect(
            rect.x - padding_x,
            rect.y - padding_y,
            rect.width + 2 * padding_x,
            rect.height + 2 * padding_y,
        )

        # Semi-translucent pill with subtle outline
        pygame.draw.rect(
            self.screen, (80, 100, 200, 220), bg_rect, border_radius=12
        )
        pygame.draw.rect(
            self.screen,
            (200, 210, 255),
            bg_rect,
            1,
            border_radius=12,
        )
        self.screen.blit(surf, rect)

    # ------------------------------------------------------------------
    # Bank & budgets drawing
    # ------------------------------------------------------------------
    def _draw_banks(self) -> None:
        """
        Draw banks in a circle with fraud heat halos, bevel, and hover feedback.
        """
        assert self.screen is not None
        if not self.bank_positions:
            return

        num_banks = len(self.bank_positions)
        current_txn = getattr(self.env, "current_transaction", None)
        bank_fraud_rates = getattr(self.env, "bank_fraud_rates", [0.05] * num_banks)

        # Determine which bank is active
        active_bank_id = None
        if current_txn is not None:
            active_bank_id = current_txn.get("bank_id")

        base_radius = int(min(self.window_width, self.window_height) * 0.055)
        mouse_x, mouse_y = pygame.mouse.get_pos()

        for i, (x, y) in enumerate(self.bank_positions):
            fraud_rate = bank_fraud_rates[i] if i < len(bank_fraud_rates) else 0.05

            # Fraud "heat" halo (multi-layer)
            heat_intensity = max(0.0, min(1.0, fraud_rate * 4))  # scale up
            self._draw_bank_halo(x, y, base_radius, heat_intensity)

            # Active bank pulse
            is_active = (active_bank_id == i)
            radius = base_radius
            if is_active:
                pulse = 1.0 + 0.08 * math.sin(self.active_bank_pulse_phase)
                radius = int(base_radius * pulse)

            # Hover highlight
            dist = math.hypot(mouse_x - x, mouse_y - y)
            is_hovered = dist < radius * 1.25

            inner_color = self.COLOR_BANK_ACTIVE if is_active else self.COLOR_BANK
            if is_hovered and not is_active:
                inner_color = (
                    min(inner_color[0] + 18, 255),
                    min(inner_color[1] + 18, 255),
                    min(inner_color[2] + 18, 255),
                )

            # Draw beveled bank circle
            self._draw_beveled_circle(x, y, radius, inner_color)

            # Bank label
            if self.font_small is not None:
                label = self.font_small.render(f"Bank {i+1}", True, self.COLOR_TEXT)
                rect = label.get_rect(center=(x, y - radius - 20))
                self.screen.blit(label, rect)

            # Fraud rate text
            if self.font_tiny is not None:
                rate_text = self.font_tiny.render(
                    f"Fraud: {fraud_rate:.1%}", True, self.COLOR_TEXT
                )
                rect = rate_text.get_rect(center=(x, y + radius + 16))
                self.screen.blit(rate_text, rect)

    def _draw_bank_halo(self, x: int, y: int, base_radius: int, heat: float) -> None:
        """Draw multi-layer fraud heat halo around bank."""
        assert self.screen is not None

        if heat <= 0.01:
            return

        # Three layers: outer, mid, inner
        radii = [
            int(base_radius * 1.45),
            int(base_radius * 1.25),
            int(base_radius * 1.1),
        ]
        alphas = [40, 70, 110]

        for r, a in zip(radii, alphas):
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            color = (
                int(255),
                int(160 + 40 * heat),
                int(140 + 20 * heat),
                int(a * heat),
            )
            pygame.draw.circle(surf, color, (r, r), r)
            self.screen.blit(surf, (x - r, y - r))

    def _draw_beveled_circle(self, x: int, y: int, radius: int, color: Color) -> None:
        """Draw a circle with a subtle bevel/light effect."""
        assert self.screen is not None

        # Base circle
        pygame.draw.circle(self.screen, color, (x, y), radius)

        # Outline
        pygame.draw.circle(self.screen, self.COLOR_BANK_OUTLINE, (x, y), radius, 2)

        # Light from top-left
        highlight_radius = int(radius * 0.75)
        highlight_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            highlight_surf,
            (255, 255, 255, 80),
            (radius - int(radius * 0.35), radius - int(radius * 0.35)),
            highlight_radius,
        )
        self.screen.blit(highlight_surf, (x - radius, y - radius))

        # Inner shading
        inner_radius = int(radius * 0.85)
        inner_surf = pygame.Surface((inner_radius * 2, inner_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(
            inner_surf,
            (0, 0, 0, 35),
            (inner_radius, inner_radius),
            inner_radius,
        )
        self.screen.blit(inner_surf, (x - inner_radius, y - inner_radius))

    def _draw_budgets(self) -> None:
        """Draw privacy and manual review budgets at the bottom-left."""
        assert self.screen is not None
        assert self.font is not None and self.font_small is not None and self.font_tiny is not None

        panel_width = int(self.window_width * 0.40)
        panel_height = int(self.window_height * 0.18)
        panel_x = int(self.window_width * 0.05)
        panel_y = int(self.window_height * 0.76)

        self._draw_card(panel_x, panel_y, panel_width, panel_height)

        title = self.font.render("Budgets", True, self.COLOR_TEXT)
        self.screen.blit(title, (panel_x + 20, panel_y + 14))

        bar_margin_left = panel_x + 110
        bar_top = panel_y + 54
        bar_width = panel_width - 150
        bar_height = 22
        line_gap = 42

        # -------- Privacy budget --------
        initial_privacy = float(getattr(self.env, "initial_privacy_budget", 1.0) or 1.0)
        used_privacy = float(getattr(self.env, "privacy_budget_used", 0.0))
        privacy_ratio = max(0.0, min(1.0, used_privacy / initial_privacy))

        label_priv = self.font_small.render("Privacy:", True, self.COLOR_TEXT)
        self.screen.blit(label_priv, (panel_x + 20, bar_top - 2))

        self._draw_gradient_bar(
            bar_margin_left,
            bar_top,
            bar_width,
            bar_height,
            base_color=self.COLOR_PRIVACY_BAR,
            ratio=privacy_ratio,
        )

        privacy_txt = self.font_tiny.render(
            f"{used_privacy:.2f} / {initial_privacy:.2f}", True, self.COLOR_TEXT
        )
        self.screen.blit(privacy_txt, (bar_margin_left + bar_width + 8, bar_top + 3))

        # -------- Manual review budget --------
        bar_top += line_gap

        initial_manual = float(getattr(self.env, "initial_manual_budget", 1.0) or 1.0)
        manual_remaining = float(getattr(self.env, "manual_budget_remaining", initial_manual))
        manual_ratio = max(0.0, min(1.0, manual_remaining / initial_manual))

        label_man = self.font_small.render("Manual:", True, self.COLOR_TEXT)
        self.screen.blit(label_man, (panel_x + 20, bar_top - 2))

        self._draw_gradient_bar(
            bar_margin_left,
            bar_top,
            bar_width,
            bar_height,
            base_color=self.COLOR_MANUAL_BAR,
            ratio=manual_ratio,
        )

        manual_txt = self.font_tiny.render(
            f"{manual_remaining:.1f} / {initial_manual:.1f}", True, self.COLOR_TEXT
        )
        self.screen.blit(manual_txt, (bar_margin_left + bar_width + 8, bar_top + 3))

    def _draw_gradient_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        base_color: Color,
        ratio: float,
    ) -> None:
        """Draw a subtle gradient fill bar with rounded corners."""
        assert self.screen is not None

        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (225, 225, 230), bg_rect, border_radius=8)

        filled_width = int(width * ratio)
        if filled_width > 0:
            fg_rect = pygame.Rect(x, y, filled_width, height)
            # gradient-like effect with simple top highlight
            pygame.draw.rect(self.screen, base_color, fg_rect, border_radius=8)
            highlight_rect = pygame.Rect(x, y, filled_width, height // 2)
            highlight = pygame.Surface(highlight_rect.size, pygame.SRCALPHA)
            highlight.fill((255, 255, 255, 50))
            self.screen.blit(highlight, highlight_rect.topleft)

        pygame.draw.rect(
            self.screen,
            self.COLOR_PANEL_BORDER,
            bg_rect,
            1,
            border_radius=8,
        )

    # ------------------------------------------------------------------
    # Right-hand panels
    # ------------------------------------------------------------------
    def _draw_transaction_panel(self) -> None:
        """Draw current transaction details on the right side."""
        assert self.screen is not None and self.font is not None and self.font_small is not None

        panel_width = int(self.window_width * 0.36)
        panel_height = int(self.window_height * 0.28)
        panel_x = int(self.window_width * 0.58)
        panel_y = int(self.window_height * 0.08)

        card_rect = self._draw_card(panel_x, panel_y, panel_width, panel_height)

        # Icon + title
        title = self.font.render("💳  Current Transaction", True, self.COLOR_TEXT)
        self.screen.blit(title, (card_rect.x + 22, card_rect.y + 14))

        txn = getattr(self.env, "current_transaction", None)
        if txn is None:
            empty_text = self.font_small.render("No transaction yet.", True, self.COLOR_TEXT)
            self.screen.blit(empty_text, (card_rect.x + 22, card_rect.y + 70))
            return

        y = card_rect.y + 70
        line_gap = 32

        bank_id = txn.get("bank_id", 0)
        amount = txn.get("amount", 0.0)
        txn_type_idx = txn.get("txn_type", 0)
        risk_score = txn.get("risk_score", 0.0)
        is_fraud = bool(txn.get("is_fraud", False))

        transaction_types = getattr(self.env, "transaction_types", ["pos", "online", "atm"])
        if 0 <= txn_type_idx < len(transaction_types):
            txn_type_str = transaction_types[txn_type_idx].title()
        else:
            txn_type_str = f"Type {txn_type_idx}"

        lines = [
            f"Bank: Bank {bank_id + 1}",
            f"Amount: ${amount:,.2f}",
            f"Type: {txn_type_str}",
            f"Risk Score: {risk_score:.2f}",
        ]

        for line in lines:
            text_surface = self.font_small.render(line, True, self.COLOR_TEXT)
            self.screen.blit(text_surface, (card_rect.x + 26, y))
            y += line_gap

        # Ground-truth label (visible to human only)
        status_text = "🚨 FRAUD" if is_fraud else "✓ LEGITIMATE"
        status_color = self.COLOR_FRAUD if is_fraud else self.COLOR_LEGIT

        status_surface = self.font.render(status_text, True, status_color)
        rect = status_surface.get_rect(
            bottomright=(card_rect.x + card_rect.width - 26, card_rect.y + card_rect.height - 20)
        )
        self.screen.blit(status_surface, rect)

    def _draw_action_panel(self) -> None:
        """Draw the agent's most recent action with a zoom/pulse animation."""
        assert self.screen is not None and self.font is not None

        panel_width = int(self.window_width * 0.36)
        panel_height = int(self.window_height * 0.16)
        panel_x = int(self.window_width * 0.58)
        panel_y = int(self.window_height * 0.38)

        card_rect = self._draw_card(panel_x, panel_y, panel_width, panel_height)

        title = self.font.render("🤖  Agent Action", True, self.COLOR_TEXT)
        self.screen.blit(title, (card_rect.x + 22, card_rect.y + 14))

        if self.last_action is None:
            return

        action_names = {0: "APPROVE", 1: "BLOCK", 2: "MANUAL REVIEW"}
        action_colors = {0: self.COLOR_APPROVE, 1: self.COLOR_BLOCK, 2: self.COLOR_REVIEW}

        name = action_names.get(self.last_action, "UNKNOWN")
        color = action_colors.get(self.last_action, self.COLOR_TEXT)

        # Pulse makes the text slightly larger
        scale = 1.0 + 0.18 * self.action_pulse
        base_font_size = 40
        size = int(base_font_size * scale)
        size = max(28, min(size, 64))

        dynamic_font = pygame.font.Font(None, size)
        text_surface = dynamic_font.render(name, True, color)
        rect = text_surface.get_rect(
            center=(card_rect.x + card_rect.width // 2, card_rect.y + card_rect.height // 2 + 4)
        )
        self.screen.blit(text_surface, rect)

    def _draw_stats_panel(self) -> None:
        """Draw episode-level statistics in a card on the right side."""
        assert self.screen is not None and self.font is not None and self.font_small is not None

        panel_width = int(self.window_width * 0.36)
        panel_height = int(self.window_height * 0.26)
        panel_x = int(self.window_width * 0.58)
        panel_y = int(self.window_height * 0.59)

        card_rect = self._draw_card(panel_x, panel_y, panel_width, panel_height)

        title = self.font.render("📊  Episode Statistics", True, self.COLOR_TEXT)
        self.screen.blit(title, (card_rect.x + 22, card_rect.y + 14))

        current_step = getattr(self.env, "current_step", 0)
        max_steps = getattr(self.env, "max_steps", 0)
        cumulative_reward = float(getattr(self.env, "cumulative_reward", 0.0))
        correct = getattr(self.env, "correct_decisions", 0)
        fraud_caught = getattr(self.env, "total_fraud_caught", 0)
        fraud_missed = getattr(self.env, "total_fraud_missed", 0)

        y = card_rect.y + 72
        line_gap = 34

        stats_lines = [
            f"Step: {current_step} / {max_steps}",
            f"Cumulative Reward: {cumulative_reward:.2f}",
            f"Correct Decisions: {correct}",
            f"Fraud Caught: {fraud_caught}",
            f"Fraud Missed: {fraud_missed}",
        ]

        for line in stats_lines:
            surf = self.font_small.render(line, True, self.COLOR_TEXT)
            self.screen.blit(surf, (card_rect.x + 26, y))
            y += line_gap

    # ------------------------------------------------------------------
    # Outcome flash overlay
    # ------------------------------------------------------------------
    def _draw_outcome_flash(self) -> None:
        """
        Draw a brief color flash overlay based on the last outcome.

        Expects env to optionally define:
            env.last_outcome in {"fraud_caught", "fraud_missed", "correct_legit"}
            env.outcome_pulse in [0, 1] (1 right after event, decays over time)
        """
        assert self.screen is not None

        outcome = getattr(self.env, "last_outcome", None)
        pulse = float(getattr(self.env, "outcome_pulse", 0.0))

        if outcome is None or pulse <= 0.01:
            return

        # Determine overlay color
        if outcome == "fraud_caught":
            base_color = (60, 220, 140)  # green-ish
        elif outcome == "fraud_missed":
            base_color = (255, 80, 80)   # red
        else:
            base_color = (120, 160, 255)  # neutral blue

        alpha = int(80 * pulse)
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((base_color[0], base_color[1], base_color[2], alpha))
        self.screen.blit(overlay, (0, 0))

    # ------------------------------------------------------------------
    # Animation state
    # ------------------------------------------------------------------
    def _update_animation_state(self) -> None:
        """Update internal animation parameters (called each frame)."""
        # Advance bank pulse
        self.active_bank_pulse_phase += self.active_bank_pulse_speed

        # Decay action pulse
        self.action_pulse *= self.action_pulse_decay
        if self.action_pulse < 0.01:
            self.action_pulse = 0.0

        # Outcome pulse decay (if env exposes it)
        if hasattr(self.env, "outcome_pulse"):
            self.env.outcome_pulse *= 0.90
            if self.env.outcome_pulse < 0.01:
                self.env.outcome_pulse = 0.0
