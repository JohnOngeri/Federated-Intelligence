"""
🎮 Game-Like Fraud Detection Visualization
Arcade-style renderer that makes RL agent behavior immediately visible and thrilling.

Features:
- Animated transaction orbs with trails
- Action gates (Approve/Block/Review)
- Consequence animations (explosions, ripples, alarms)
- Animated bank nodes with pulsing/glowing
- Scrolling timeline bar
- Threat meter
- Agent brain avatar
- Real-time probability bars
- Slow-motion replay for critical decisions
- Sound effects (optional)
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional, Tuple, List, Dict
from dataclasses import dataclass
from enum import Enum

import pygame
import numpy as np

Color = Tuple[int, int, int]


class OrbState(Enum):
    """States for transaction orbs"""
    SPAWNING = "spawning"
    TRAVELING = "traveling"
    APPROACHING = "approaching"
    DECIDING = "deciding"
    CONSEQUENCE = "consequence"
    FADING = "fading"


class ConsequenceType(Enum):
    """Types of consequence animations"""
    FRAUD_CAUGHT = "fraud_caught"
    FRAUD_MISSED = "fraud_missed"
    LEGIT_BLOCKED = "legit_blocked"
    LEGIT_APPROVED = "legit_approved"


@dataclass
class Orb:
    """Represents an animated transaction orb"""
    x: float
    y: float
    target_x: float
    target_y: float
    radius: float
    color: Color
    state: OrbState
    progress: float  # 0.0 to 1.0 along path
    is_fraud: bool
    amount: float
    bank_id: int
    age: float  # Time since spawn
    trail: List[Tuple[float, float]]  # Previous positions
    rotation: float  # Rotation angle
    glow_intensity: float  # 0.0 to 1.0
    consequence: Optional[ConsequenceType] = None
    consequence_age: float = 0.0


@dataclass
class Particle:
    """Particle for effects"""
    x: float
    y: float
    vx: float
    vy: float
    color: Color
    size: float
    life: float  # 0.0 to 1.0
    decay: float


class GameRenderer:
    """Arcade-style game-like visualization for fraud detection"""
    
    def __init__(
        self,
        env: Any,
        window_size: Tuple[int, int] = (1600, 900),
        enable_sound: bool = False,
    ) -> None:
        """
        Initialize the game renderer.
        
        Args:
            env: The PrivFedFraudEnv instance
            window_size: Window dimensions
            enable_sound: Whether to enable sound effects
        """
        self.env = env
        self.window_width, self.window_height = window_size
        
        # Layout system
        self.scale_factor = 1.0
        self.safe_margin = 20
        self.ui_padding = 10
        self.layout_zones = {}
        self.anchor_points = {}
        
        # Pygame setup
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None
        
        # Game state
        self.orbs: List[Orb] = []
        self.particles: List[Particle] = []
        self.timeline_dots: List[Dict] = []  # History of decisions
        self.current_action: Optional[int] = None
        self.current_action_probs: Dict[int, float] = {0: 0.33, 1: 0.33, 2: 0.34}
        self.slow_motion_active: bool = False
        self.slow_motion_timer: float = 0.0
        
        # Animation state
        self.time: float = 0.0
        self.last_transaction_time: float = 0.0
        self.transaction_interval: float = 1.5  # Seconds between transactions
        
        # Bank positions (circular layout)
        self.bank_positions: List[Tuple[int, int]] = []
        self.bank_pulse_phase: List[float] = [0.0] * 5
        
        # Gate positions
        self.gate_positions: List[Tuple[int, int]] = []
        self.gate_glow: List[float] = [0.0, 0.0, 0.0]  # Approve, Block, Review
        
        # Agent brain state
        self.brain_glow: float = 0.0
        self.brain_color: Color = (100, 200, 255)
        self.neuron_sparks: List[Dict] = []
        
        # Threat meter state
        self.threat_level: float = 0.0  # 0.0 to 1.0
        self.fraud_caught_count: int = 0
        self.fraud_missed_count: int = 0
        
        # Colors - Arcade style
        self.COLOR_BG = (10, 10, 20)  # Dark background
        self.COLOR_ORB_LEGIT = (0, 200, 255)  # Bright cyan
        self.COLOR_ORB_FRAUD = (255, 0, 100)  # Bright red
        self.COLOR_ORB_UNCERTAIN = (255, 200, 0)  # Yellow
        self.COLOR_GATE_APPROVE = (0, 255, 136)  # Green
        self.COLOR_GATE_BLOCK = (255, 0, 68)  # Red
        self.COLOR_GATE_REVIEW = (255, 170, 0)  # Yellow
        self.COLOR_BANK_LOW = (100, 150, 255)  # Blue
        self.COLOR_BANK_HIGH = (255, 100, 100)  # Red
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_TEXT_BRIGHT = (255, 255, 0)
        
        # Sound system
        self.enable_sound = enable_sound
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.sound_volume = 0.5
        
        # Bank sprites
        self.bank_sprites: List[Optional[pygame.Surface]] = []
        self.bank_sprite_size = (80, 100)
        self.window_flicker_timers: List[float] = [0.0] * 5
        
        # UI flags
        self.show_debug = False
        self._should_quit = False
        
        self._init_pygame()
        self._load_bank_sprites()
        self._calculate_layout()
        self._init_sounds()
    
    def _init_pygame(self) -> None:
        """Initialize pygame"""
        pygame.init()
        pygame.display.set_caption("🎮 Fraud Detection Arena - RL Agent Demo")
        
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height),
            pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 20)
        except:
            self.font_large = pygame.font.SysFont("arial", 48)
            self.font = pygame.font.SysFont("arial", 32)
            self.font_small = pygame.font.SysFont("arial", 20)
    
    def _init_sounds(self) -> None:
        """Initialize sound system"""
        if not self.enable_sound:
            return
        
        try:
            pygame.mixer.init()
            # Note: Sound files would need to be added to a sounds/ directory
            # For now, we'll just set up the structure
            sound_files = {
                "approve": None,  # "sounds/ding.wav"
                "block": None,     # "sounds/buzz.wav"
                "review": None,    # "sounds/whoosh.wav"
                "fraud_caught": None,  # "sounds/boom.wav"
                "fraud_missed": None,  # "sounds/alarm.wav"
                "legit_blocked": None,  # "sounds/thud.wav"
                "spawn": None,     # "sounds/pop.wav"
            }
            
            for name, path in sound_files.items():
                if path:
                    try:
                        self.sounds[name] = pygame.mixer.Sound(path)
                        self.sounds[name].set_volume(self.sound_volume)
                    except:
                        self.sounds[name] = None
                else:
                    self.sounds[name] = None
        except:
            pass  # Sound system optional
    
    def _play_sound(self, name: str) -> None:
        """Play a sound effect"""
        if not self.enable_sound or name not in self.sounds:
            return
        if self.sounds[name]:
            try:
                self.sounds[name].play()
            except:
                pass
    
    def _calculate_layout(self) -> None:
        """Calculate responsive layout with safe areas and dynamic anchoring"""
        if self.screen is None:
            return
        
        self.window_width, self.window_height = self.screen.get_size()
        
        # Calculate scale factor for retina displays
        base_width, base_height = 1600, 900
        self.scale_factor = min(self.window_width / base_width, self.window_height / base_height)
        
        # Define safe area (with margins)
        safe_left = self.safe_margin
        safe_right = self.window_width - self.safe_margin
        safe_top = self.safe_margin
        safe_bottom = self.window_height - self.safe_margin - 80  # Reserve space for timeline
        
        safe_width = safe_right - safe_left
        safe_height = safe_bottom - safe_top
        
        # Define layout zones with auto-padding
        self.layout_zones = {
            'banks': {
                'x': safe_left,
                'y': safe_top,
                'width': int(safe_width * 0.25),
                'height': safe_height,
                'center_x': safe_left + int(safe_width * 0.125),
                'center_y': safe_top + safe_height // 2
            },
            'gates': {
                'x': safe_left + int(safe_width * 0.35),
                'y': safe_top,
                'width': int(safe_width * 0.3),
                'height': safe_height,
                'center_x': safe_left + int(safe_width * 0.5),
                'center_y': safe_top + safe_height // 2
            },
            'stats': {
                'x': safe_right - int(safe_width * 0.25),
                'y': safe_top,
                'width': int(safe_width * 0.25),
                'height': safe_height,
                'center_x': safe_right - int(safe_width * 0.125),
                'center_y': safe_top + safe_height // 2
            },
            'timeline': {
                'x': safe_left,
                'y': safe_bottom + self.ui_padding,
                'width': safe_width,
                'height': 50,
                'center_x': safe_left + safe_width // 2,
                'center_y': safe_bottom + self.ui_padding + 25
            }
        }
        
        # Calculate bank positions (vertical column layout)
        num_banks = getattr(self.env, "num_banks", 5)
        bank_zone = self.layout_zones['banks']
        
        self.bank_positions.clear()
        vertical_spacing = bank_zone['height'] // (num_banks + 1)
        start_y = bank_zone['y'] + vertical_spacing
        
        for i in range(num_banks):
            x = bank_zone['center_x']
            y = start_y + i * vertical_spacing
            self.bank_positions.append((x, y))
        
        # Calculate gate positions (perfectly aligned horizontal layout)
        gate_zone = self.layout_zones['gates']
        gate_spacing = gate_zone['width'] // 3
        gate_start_x = gate_zone['x'] + gate_spacing // 2
        
        self.gate_positions = [
            (gate_start_x, gate_zone['center_y']),
            (gate_start_x + gate_spacing, gate_zone['center_y']),
            (gate_start_x + gate_spacing * 2, gate_zone['center_y'])
        ]
        
        # Define anchor points for UI elements (grid-aligned)
        stats_zone = self.layout_zones['stats']
        gate_zone = self.layout_zones['gates']
        spacing = int(180 * self.scale_factor)
        
        self.anchor_points = {
            'brain': (gate_zone['center_x'], gate_zone['y'] + int(60 * self.scale_factor)),
            'prob_bars': (stats_zone['x'] + self.ui_padding, stats_zone['y'] + self.ui_padding),
            'threat_meter': (stats_zone['center_x'], stats_zone['y'] + spacing),
            'stats_text': (stats_zone['x'] + self.ui_padding, stats_zone['y'] + spacing * 2)
        }
    
    def set_action(self, action: int, action_probs: Optional[Dict[int, float]] = None) -> None:
        """Set the current action and probabilities"""
        self.current_action = action
        if action_probs:
            self.current_action_probs = action_probs
        else:
            # Default probabilities if not provided
            self.current_action_probs = {0: 0.0, 1: 0.0, 2: 0.0}
            self.current_action_probs[action] = 1.0
    
    def _spawn_orb(self, bank_id: int, is_fraud: bool, amount: float, risk_score: float) -> None:
        """Spawn a new transaction orb"""
        if bank_id >= len(self.bank_positions):
            return
        
        bank_x, bank_y = self.bank_positions[bank_id]
        
        # Set target to current action gate immediately
        action_to_use = self.current_action if self.current_action is not None else int(self.time) % 3
        target_x, target_y = self.gate_positions[action_to_use]
        
        # Determine orb color
        if is_fraud:
            base_color = self.COLOR_ORB_FRAUD
        elif risk_score > 0.4 and risk_score < 0.7:
            base_color = self.COLOR_ORB_UNCERTAIN
        else:
            base_color = self.COLOR_ORB_LEGIT
        
        # Orb size based on amount with scaling
        base_radius = 15 + (amount / 5000.0) * 25  # 15-40px
        radius = base_radius * self.scale_factor
        
        orb = Orb(
            x=float(bank_x),
            y=float(bank_y),
            target_x=target_x,
            target_y=target_y,
            radius=radius,
            color=base_color,
            state=OrbState.SPAWNING,
            progress=0.0,
            is_fraud=is_fraud,
            amount=amount,
            bank_id=bank_id,
            age=0.0,
            trail=[],
            rotation=0.0,
            glow_intensity=0.0,
        )
        
        self.orbs.append(orb)
        self._play_sound("spawn")
        
        # Bank pulse effect
        self.bank_pulse_phase[bank_id] = 1.0
    
    def _update_orbs(self, dt: float) -> None:
        """Update all orbs"""
        speed_multiplier = 0.3 if self.slow_motion_active else 1.0
        
        for orb in self.orbs[:]:
            orb.age += dt
            
            # Update state
            if orb.state == OrbState.SPAWNING:
                if orb.age > 0.2:
                    orb.state = OrbState.TRAVELING
                    orb.glow_intensity = 0.5
            
            elif orb.state == OrbState.TRAVELING:
                # Move along fixed path (slower for visibility)
                travel_speed = 0.8 * speed_multiplier
                orb.progress += dt * travel_speed
                
                if orb.progress >= 1.0:
                    orb.progress = 1.0
                    orb.state = OrbState.APPROACHING
                
                # Curved arc to target gate
                start_x, start_y = self.bank_positions[orb.bank_id]
                
                t = orb.progress
                # Bezier curve with mid-point for natural curvature
                mid_x = (start_x + orb.target_x) / 2
                mid_y = min(start_y, orb.target_y) - 120 * self.scale_factor
                
                orb.x = (1-t)**2 * start_x + 2*(1-t)*t * mid_x + t**2 * orb.target_x
                orb.y = (1-t)**2 * start_y + 2*(1-t)*t * mid_y + t**2 * orb.target_y
                
                # Update trail
                orb.trail.append((orb.x, orb.y))
                if len(orb.trail) > 10:
                    orb.trail.pop(0)
                
                # Update rotation
                orb.rotation += dt * 3.0
                
                # Update glow
                orb.glow_intensity = 0.5 + 0.5 * math.sin(orb.age * 5.0)
            
            elif orb.state == OrbState.APPROACHING:
                # Slow down near gate
                approach_speed = 0.5 * speed_multiplier
                orb.progress = min(1.0, orb.progress + dt * approach_speed)
                
                if orb.progress >= 1.0:
                    orb.state = OrbState.DECIDING
                    orb.progress = 0.0  # Reset progress for deciding phase
            
            elif orb.state == OrbState.DECIDING:
                # Orb should already be at target, just finish
                orb.state = OrbState.CONSEQUENCE
                orb.consequence_age = 0.0
                
                # Determine which gate was hit
                action_used = 0
                for i, (gx, gy) in enumerate(self.gate_positions):
                    if abs(orb.target_x - gx) < 10 and abs(orb.target_y - gy) < 10:
                        action_used = i
                        break
                
                self.gate_glow[action_used] = 1.0
                
                if True:  # Always trigger consequence
                    orb.state = OrbState.CONSEQUENCE
                    orb.consequence_age = 0.0
                    # Determine consequence
                    if orb.is_fraud:
                        if self.current_action == 1:  # BLOCK
                            orb.consequence = ConsequenceType.FRAUD_CAUGHT
                            self._play_sound("fraud_caught")
                            self.fraud_caught_count += 1
                        elif self.current_action == 2:  # REVIEW
                            orb.consequence = ConsequenceType.FRAUD_CAUGHT
                            self._play_sound("fraud_caught")
                            self.fraud_caught_count += 1
                        else:  # APPROVE
                            orb.consequence = ConsequenceType.FRAUD_MISSED
                            self._play_sound("fraud_missed")
                            self.fraud_missed_count += 1
                    else:
                        if self.current_action == 1:  # BLOCK
                            orb.consequence = ConsequenceType.LEGIT_BLOCKED
                            self._play_sound("legit_blocked")
                        else:  # APPROVE or REVIEW
                            orb.consequence = ConsequenceType.LEGIT_APPROVED
                            self._play_sound("approve")
                else:
                    # Smooth curved path to gate
                    orb.x += dx * move_speed * dt
                    orb.y += dy * move_speed * dt
            
            elif orb.state == OrbState.CONSEQUENCE:
                orb.consequence_age += dt
                
                # Create particles for consequence
                if orb.consequence_age < 0.1:  # Only once
                    self._create_consequence_particles(orb)
                
                if orb.consequence_age > 0.5:
                    orb.state = OrbState.FADING
            
            elif orb.state == OrbState.FADING:
                orb.glow_intensity -= dt * 2.0
                if orb.glow_intensity <= 0.0:
                    # Add to timeline
                    self._add_timeline_dot(orb)
                    self.orbs.remove(orb)
                    continue
        
        # Update bank pulse
        for i in range(len(self.bank_pulse_phase)):
            if self.bank_pulse_phase[i] > 0.0:
                self.bank_pulse_phase[i] -= dt * 2.0
                self.bank_pulse_phase[i] = max(0.0, self.bank_pulse_phase[i])
    
    def _create_consequence_particles(self, orb: Orb) -> None:
        """Create particles for consequence animation"""
        num_particles = 20
        
        if orb.consequence == ConsequenceType.FRAUD_CAUGHT:
            # Red explosion
            color = self.COLOR_ORB_FRAUD
            for _ in range(num_particles):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(100, 300)
                self.particles.append(Particle(
                    x=orb.x, y=orb.y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=color,
                    size=random.uniform(3, 8),
                    life=1.0,
                    decay=random.uniform(0.5, 1.5)
                ))
        
        elif orb.consequence == ConsequenceType.FRAUD_MISSED:
            # Red alarm particles
            color = (255, 0, 0)
            for _ in range(num_particles // 2):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(50, 150)
                self.particles.append(Particle(
                    x=orb.x, y=orb.y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=color,
                    size=random.uniform(2, 5),
                    life=1.0,
                    decay=0.8
                ))
        
        elif orb.consequence == ConsequenceType.LEGIT_BLOCKED:
            # Blue ripple
            color = self.COLOR_ORB_LEGIT
            for _ in range(num_particles // 2):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(30, 80)
                self.particles.append(Particle(
                    x=orb.x, y=orb.y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=color,
                    size=random.uniform(2, 6),
                    life=1.0,
                    decay=1.2
                ))
        
        elif orb.consequence == ConsequenceType.LEGIT_APPROVED:
            # Green sparkles
            color = (0, 255, 136)
            for _ in range(num_particles // 3):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(20, 60)
                self.particles.append(Particle(
                    x=orb.x, y=orb.y,
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=color,
                    size=random.uniform(1, 4),
                    life=1.0,
                    decay=2.0
                ))
    
    def _add_timeline_dot(self, orb: Orb) -> None:
        """Add a dot to the timeline"""
        # Determine dot color
        if orb.consequence == ConsequenceType.FRAUD_CAUGHT or \
           orb.consequence == ConsequenceType.LEGIT_APPROVED:
            color = (0, 255, 0)  # Green (correct)
        elif orb.consequence == ConsequenceType.FRAUD_MISSED or \
             orb.consequence == ConsequenceType.LEGIT_BLOCKED:
            color = (255, 0, 0)  # Red (mistake)
        else:
            color = (255, 255, 0)  # Yellow (review)
        
        self.timeline_dots.append({
            "color": color,
            "age": 0.0,
            "x": self.window_width - 20,
        })
    
    def _update_particles(self, dt: float) -> None:
        """Update all particles"""
        for particle in self.particles[:]:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.life -= particle.decay * dt
            
            if particle.life <= 0.0:
                self.particles.remove(particle)
    
    def _update_timeline(self, dt: float) -> None:
        """Update timeline dots"""
        scroll_speed = 50.0  # pixels per second
        
        for dot in self.timeline_dots[:]:
            dot["x"] -= scroll_speed * dt
            dot["age"] += dt
            
            if dot["x"] < -20:
                self.timeline_dots.remove(dot)
    
    def _update_threat_meter(self) -> None:
        """Update threat meter based on stats"""
        info = self.env._get_info() if hasattr(self.env, "_get_info") else {}
        fraud_caught = info.get("fraud_caught", 0)
        fraud_missed = info.get("fraud_missed", 0)
        total = fraud_caught + fraud_missed
        
        if total > 0:
            # Threat increases with missed fraud
            self.threat_level = min(1.0, fraud_missed / max(1, total * 0.3))
        else:
            self.threat_level = 0.0
    
    def _update_brain(self, dt: float) -> None:
        """Update agent brain animation"""
        # Brain glows when processing
        if len(self.orbs) > 0 and any(o.state in [OrbState.TRAVELING, OrbState.APPROACHING] for o in self.orbs):
            self.brain_glow = min(1.0, self.brain_glow + dt * 3.0)
            
            # Determine brain color based on confidence
            max_prob = max(self.current_action_probs.values()) if self.current_action_probs else 0.5
            if max_prob > 0.8:
                self.brain_color = (0, 255, 136)  # Green (high confidence)
            elif max_prob > 0.5:
                self.brain_color = (255, 200, 0)  # Yellow (medium)
            else:
                self.brain_color = (255, 100, 100)  # Red (low)
        else:
            self.brain_glow = max(0.0, self.brain_glow - dt * 2.0)
        
        # Update gate glow
        for i in range(len(self.gate_glow)):
            self.gate_glow[i] = max(0.0, self.gate_glow[i] - dt * 2.0)
    
    def _draw_background(self) -> None:
        """Draw background"""
        if self.screen is None:
            return
        
        # Dark background
        self.screen.fill(self.COLOR_BG)
        
        # Subtle grid pattern
        grid_size = 50
        for x in range(0, self.window_width, grid_size):
            pygame.draw.line(self.screen, (20, 20, 30), (x, 0), (x, self.window_height), 1)
        for y in range(0, self.window_height, grid_size):
            pygame.draw.line(self.screen, (20, 20, 30), (0, y), (self.window_width, y), 1)
    
    def _load_bank_sprites(self) -> None:
        """Load bank building sprites"""
        sprite_paths = [
            "assets/bank1.png", "assets/bank2.png", "assets/bank3.png",
            "assets/bank4.png", "assets/bank5.png"
        ]
        
        for path in sprite_paths:
            try:
                sprite = pygame.image.load(path).convert_alpha()
                self.bank_sprites.append(sprite)
            except:
                # Fallback: create 3D building sprite
                sprite = pygame.Surface(self.bank_sprite_size, pygame.SRCALPHA)
                # Shadow (depth)
                pygame.draw.rect(sprite, (40, 60, 80), (15, 25, 60, 70))
                # Main building
                pygame.draw.rect(sprite, (80, 120, 180), (10, 20, 60, 70))
                # Rooftop detail
                pygame.draw.rect(sprite, (60, 90, 140), (10, 20, 60, 3))
                # Windows with glow
                for row in range(3):
                    for col in range(3):
                        wx, wy = 15 + col * 18, 25 + row * 20
                        pygame.draw.rect(sprite, (200, 200, 50), (wx, wy, 8, 12))
                self.bank_sprites.append(sprite)
    
    def _draw_banks(self) -> None:
        """Draw animated bank building sprites"""
        if self.screen is None:
            return
        
        num_banks = len(self.bank_positions)
        bank_fraud_rates = getattr(self.env, "bank_fraud_rates", [0.05] * num_banks)
        
        for i, (x, y) in enumerate(self.bank_positions):
            if i >= len(self.bank_sprites):
                continue
                
            # Pulse and shake effects
            pulse = 1.0 + self.bank_pulse_phase[i] * 0.2 + 0.05 * math.sin(self.time * 2.0 + i)
            fraud_rate = bank_fraud_rates[i] if i < len(bank_fraud_rates) else 0.1
            
            shake_x = shake_y = 0
            if fraud_rate > 0.15:
                shake_x = random.uniform(-1, 1) if self.time % 0.3 < 0.05 else 0
                shake_y = random.uniform(-1, 1) if self.time % 0.3 < 0.05 else 0
            
            # Scale sprite
            scaled_size = (int(self.bank_sprite_size[0] * pulse * self.scale_factor),
                          int(self.bank_sprite_size[1] * pulse * self.scale_factor))
            scaled_sprite = pygame.transform.scale(self.bank_sprites[i], scaled_size)
            
            # Window flicker animation with brightness variation
            flicker_sprite = scaled_sprite.copy()
            for row in range(3):
                for col in range(3):
                    if random.random() < 0.3:  # 30% chance to flicker
                        brightness = random.randint(200, 255)
                        fx = int(15 + col * 18 * pulse * self.scale_factor)
                        fy = int(25 + row * 20 * pulse * self.scale_factor)
                        fw, fh = int(8 * pulse * self.scale_factor), int(12 * pulse * self.scale_factor)
                        pygame.draw.rect(flicker_sprite, (brightness, brightness, brightness//2), (fx, fy, fw, fh))
            scaled_sprite = flicker_sprite
            
            # Glow effect from sprite edges
            glow_color = self.COLOR_BANK_HIGH if fraud_rate > 0.15 else self.COLOR_BANK_LOW
            glow_alpha = int(80 * (0.5 + 0.5 * math.sin(self.time * 3.0 + i)))
            glow_size = (scaled_size[0] + 20, scaled_size[1] + 20)
            glow_surf = pygame.Surface(glow_size, pygame.SRCALPHA)
            
            # Create edge glow
            for offset in range(5):
                alpha = glow_alpha // (offset + 1)
                glow_rect = pygame.Rect(10 - offset, 10 - offset, 
                                       scaled_size[0] + offset * 2, scaled_size[1] + offset * 2)
                pygame.draw.rect(glow_surf, (*glow_color, alpha), glow_rect, 1)
            
            # Position and draw
            final_x = int(x + shake_x - scaled_size[0] // 2)
            final_y = int(y + shake_y - scaled_size[1] // 2)
            
            self.screen.blit(glow_surf, (final_x - 10, final_y - 10))
            self.screen.blit(scaled_sprite, (final_x, final_y))
            
            # Bank ID label
            text = self.font_small.render(f"B{i+1}", True, self.COLOR_TEXT)
            text_rect = text.get_rect(center=(x, y + scaled_size[1] // 2 + 15))
            self.screen.blit(text, text_rect)
    
    def _draw_gates(self) -> None:
        """Draw action gates"""
        if self.screen is None:
            return
        
        gate_names = ["APPROVE", "BLOCK", "REVIEW"]
        gate_colors = [self.COLOR_GATE_APPROVE, self.COLOR_GATE_BLOCK, self.COLOR_GATE_REVIEW]
        gate_icons = ["✓", "✗", "?"]
        
        for i, ((x, y), name, color, icon) in enumerate(zip(self.gate_positions, gate_names, gate_colors, gate_icons)):
            # Gate glow
            glow = self.gate_glow[i]
            if glow > 0.0:
                glow_radius = int(60 * (1.0 + glow * 0.5))
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                glow_alpha = int(150 * glow)
                pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (x - glow_radius, y - glow_radius))
            
            # Gate archway with responsive sizing
            gate_width = int(80 * self.scale_factor)
            gate_height = int(120 * self.scale_factor)
            
            # Draw arch
            rect = pygame.Rect(x - gate_width // 2, y - gate_height // 2, gate_width, gate_height)
            pygame.draw.rect(self.screen, color, rect, 3)
            
            # Icon
            icon_text = self.font.render(icon, True, color)
            icon_rect = icon_text.get_rect(center=(x, y - 20))
            self.screen.blit(icon_text, icon_rect)
            
            # Label
            label = self.font_small.render(name, True, color)
            label_rect = label.get_rect(center=(x, y + 30))
            self.screen.blit(label, label_rect)
            
            # Probability bar width (if action selected)
            if self.current_action == i and self.current_action_probs:
                prob = self.current_action_probs.get(i, 0.0)
                bar_width = int(gate_width * prob)
                bar_rect = pygame.Rect(x - bar_width // 2, y + 50, bar_width, 5)
                pygame.draw.rect(self.screen, color, bar_rect)
    
    def _draw_orbs(self) -> None:
        """Draw transaction orbs with trajectory lines and arrows"""
        if self.screen is None:
            return
        
        for orb in self.orbs:
            # Draw trajectory curve line
            if orb.state == OrbState.TRAVELING:
                start_x, start_y = self.bank_positions[orb.bank_id]
                mid_x = (start_x + orb.target_x) / 2
                mid_y = min(start_y, orb.target_y) - 120 * self.scale_factor
                
                # Draw bezier curve path
                points = []
                for t in np.linspace(0, 1, 20):
                    x = (1-t)**2 * start_x + 2*(1-t)*t * mid_x + t**2 * orb.target_x
                    y = (1-t)**2 * start_y + 2*(1-t)*t * mid_y + t**2 * orb.target_y
                    points.append((int(x), int(y)))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, (255, 255, 255, 100), False, points, 2)
                
                # Draw arrow at target
                arrow_size = 15
                pygame.draw.polygon(self.screen, orb.color, [
                    (int(orb.target_x), int(orb.target_y - arrow_size)),
                    (int(orb.target_x - arrow_size//2), int(orb.target_y)),
                    (int(orb.target_x + arrow_size//2), int(orb.target_y))
                ])
            
            # Draw trail
            for i, (tx, ty) in enumerate(orb.trail):
                alpha = i / max(len(orb.trail), 1)
                trail_color = (*orb.color, int(100 * alpha))
                trail_surf = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, trail_color, (5, 5), 3)
                self.screen.blit(trail_surf, (int(tx - 5), int(ty - 5)))
            
            # Draw orb glow
            if orb.glow_intensity > 0.0:
                glow_radius = int(orb.radius * (1.0 + orb.glow_intensity * 0.5))
                glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                glow_alpha = int(150 * orb.glow_intensity)
                pygame.draw.circle(glow_surf, (*orb.color, glow_alpha), (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surf, (int(orb.x - glow_radius), int(orb.y - glow_radius)))
            
            # Draw orb
            pygame.draw.circle(self.screen, orb.color, (int(orb.x), int(orb.y)), int(orb.radius))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(orb.x), int(orb.y)), int(orb.radius), 2)
            
            # Draw amount (if large enough)
            if orb.radius > 20 * self.scale_factor:
                amount_text = self.font_small.render(f"${orb.amount:.0f}", True, self.COLOR_TEXT)
                amount_rect = amount_text.get_rect(center=(int(orb.x), int(orb.y)))
                self.screen.blit(amount_text, amount_rect)
    
    def _draw_particles(self) -> None:
        """Draw particles"""
        if self.screen is None:
            return
        
        for particle in self.particles:
            alpha = int(255 * particle.life)
            size = int(particle.size * particle.life)
            if size > 0:
                color_with_alpha = (*particle.color, alpha)
                surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color_with_alpha, (size, size), size)
                self.screen.blit(surf, (int(particle.x - size), int(particle.y - size)))
    
    def _draw_agent_brain(self) -> None:
        """Draw agent brain avatar with responsive positioning"""
        if self.screen is None or 'brain' not in self.anchor_points:
            return
        
        brain_x, brain_y = self.anchor_points['brain']
        brain_size = int(36 * self.scale_factor)  # 40% smaller
        
        # Softer glow with neural pulse
        if self.brain_glow > 0.0:
            pulse_factor = 0.8 + 0.2 * math.sin(self.time * 4.0)
            glow_radius = int(brain_size * (1.0 + self.brain_glow * 0.3 * pulse_factor))
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            glow_alpha = int(80 * self.brain_glow * pulse_factor)  # Much softer
            pygame.draw.circle(glow_surf, (*self.brain_color, glow_alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surf, (brain_x - glow_radius, brain_y - glow_radius))
        
        # Brain circle
        pygame.draw.circle(self.screen, self.brain_color, (brain_x, brain_y), brain_size)
        pygame.draw.circle(self.screen, (255, 255, 255), (brain_x, brain_y), brain_size, 2)
        
        # Brain icon (simple representation)
        # Draw neural network nodes
        node_positions = [
            (brain_x - 15, brain_y - 15),
            (brain_x + 15, brain_y - 15),
            (brain_x, brain_y),
            (brain_x - 15, brain_y + 15),
            (brain_x + 15, brain_y + 15),
        ]
        
        for nx, ny in node_positions:
            pygame.draw.circle(self.screen, (255, 255, 255), (nx, ny), 3)
        
        # Connections (animated)
        if self.brain_glow > 0.5:
            for i in range(len(node_positions) - 1):
                pygame.draw.line(self.screen, self.brain_color, node_positions[i], node_positions[i+1], 1)
        
        # Label
        label = self.font_small.render("AI", True, self.COLOR_TEXT)
        label_rect = label.get_rect(center=(brain_x, brain_y + brain_size + 15))
        self.screen.blit(label, label_rect)
    
    def _draw_probability_bars(self) -> None:
        """Draw real-time probability bars with responsive layout"""
        if self.screen is None or 'prob_bars' not in self.anchor_points:
            return
        
        bar_x, bar_y = self.anchor_points['prob_bars']
        bar_width = int(40 * self.scale_factor)
        bar_height = int(150 * self.scale_factor)
        bar_spacing = int(70 * self.scale_factor)
        
        action_names = ["APPROVE", "BLOCK", "REVIEW"]
        action_colors = [self.COLOR_GATE_APPROVE, self.COLOR_GATE_BLOCK, self.COLOR_GATE_REVIEW]
        
        for i, (name, color) in enumerate(zip(action_names, action_colors)):
            x = bar_x
            y = bar_y + i * bar_spacing
            
            # Stroked background frame
            frame_rect = pygame.Rect(x - 5, y - 5, bar_width + 10, bar_height + 10)
            pygame.draw.rect(self.screen, (60, 60, 60), frame_rect, 2)
            
            # Background
            bg_rect = pygame.Rect(x, y, bar_width, bar_height)
            pygame.draw.rect(self.screen, (40, 40, 40), bg_rect)
            
            # Fill bar
            prob = self.current_action_probs.get(i, 0.0)
            fill_height = int(bar_height * prob)
            if fill_height > 0:
                fill_rect = pygame.Rect(x, y + bar_height - fill_height, bar_width, fill_height)
                pygame.draw.rect(self.screen, color, fill_rect)
                
                # Glow if selected
                if self.current_action == i:
                    glow_rect = pygame.Rect(x - 2, y + bar_height - fill_height - 2, bar_width + 4, fill_height + 4)
                    pygame.draw.rect(self.screen, color, glow_rect, 2)
            
            # Label and percentage on left side
            label_text = f"{name[:4]} {int(prob * 100)}%"
            label = self.font_small.render(label_text, True, color)
            self.screen.blit(label, (x - 80, y + bar_height // 2 - 10))
    
    def _draw_threat_meter(self) -> None:
        """Draw threat meter with responsive positioning"""
        if self.screen is None or 'threat_meter' not in self.anchor_points:
            return
        
        meter_x, meter_y = self.anchor_points['threat_meter']
        meter_width = int(40 * self.scale_factor)
        meter_height = int(200 * self.scale_factor)
        meter_x -= meter_width // 2  # Center on anchor point
        
        # Background
        bg_rect = pygame.Rect(meter_x, meter_y, meter_width, meter_height)
        pygame.draw.rect(self.screen, (40, 40, 40), bg_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), bg_rect, 2)
        
        # Threat level (red from top)
        threat_height = int(meter_height * self.threat_level)
        if threat_height > 0:
            threat_rect = pygame.Rect(meter_x, meter_y, meter_width, threat_height)
            pygame.draw.rect(self.screen, (255, 0, 0), threat_rect)
        
        # Fraud caught (green from bottom)
        info = self.env._get_info() if hasattr(self.env, "_get_info") else {}
        fraud_caught = info.get("fraud_caught", 0)
        max_caught = 20  # Scale
        caught_height = int(meter_height * min(1.0, fraud_caught / max_caught))
        if caught_height > 0:
            caught_rect = pygame.Rect(meter_x, meter_y + meter_height - caught_height, meter_width, caught_height)
            pygame.draw.rect(self.screen, (0, 255, 0), caught_rect)
        
        # Label
        label = self.font_small.render("THREAT", True, self.COLOR_TEXT)
        label_rect = label.get_rect(center=(meter_x + meter_width // 2, meter_y - 20))
        self.screen.blit(label, label_rect)
    
    def _draw_timeline(self) -> None:
        """Draw scrolling timeline bar in safe area"""
        if self.screen is None or 'timeline' not in self.layout_zones:
            return
        
        timeline_zone = self.layout_zones['timeline']
        timeline_height = int(30 * self.scale_factor)
        
        # Background
        bg_rect = pygame.Rect(timeline_zone['x'], timeline_zone['y'], timeline_zone['width'], timeline_height)
        pygame.draw.rect(self.screen, (20, 20, 30), bg_rect)
        pygame.draw.rect(self.screen, (60, 60, 80), bg_rect, 2)
        
        # Draw dots
        for dot in self.timeline_dots:
            alpha = 1.0 - min(1.0, dot["age"] / 5.0)  # Fade over 5 seconds
            dot_radius = int(8 * self.scale_factor)
            dot_color = tuple(int(c * alpha) for c in dot["color"])
            pygame.draw.circle(self.screen, dot_color, (int(dot["x"]), timeline_zone['y'] + timeline_height // 2), dot_radius)
        
        # Label
        label = self.font_small.render("HISTORY", True, self.COLOR_TEXT)
        self.screen.blit(label, (timeline_zone['x'] + self.ui_padding, timeline_zone['y'] - int(25 * self.scale_factor)))
    
    def _draw_stats(self) -> None:
        """Draw statistics with responsive positioning"""
        if self.screen is None or 'stats_text' not in self.anchor_points:
            return
        
        info = self.env._get_info() if hasattr(self.env, "_get_info") else {}
        stats_x, stats_y = self.anchor_points['stats_text']
        line_height = int(25 * self.scale_factor)
        
        stats = [
            f"Step: {info.get('step', 0)}",
            f"Reward: {info.get('cumulative_reward', 0):.1f}",
            f"Fraud Caught: {info.get('fraud_caught', 0)}",
            f"Fraud Missed: {info.get('fraud_missed', 0)}",
            f"Accuracy: {info.get('correct_decisions', 0) / max(1, info.get('step', 1)) * 100:.1f}%",
        ]
        
        y_offset = 0
        for stat in stats:
            text = self.font_small.render(stat, True, self.COLOR_TEXT)
            self.screen.blit(text, (stats_x, stats_y + y_offset))
            y_offset += line_height
        
        # Fraud & Review Log
        y_offset += line_height
        header = self.font_small.render("FRAUD ALERTS:", True, (255, 100, 100))
        self.screen.blit(header, (stats_x, stats_y + y_offset))
        y_offset += line_height
        
        # Show recent fraud cases
        if hasattr(self, 'fraud_log'):
            for entry in self.fraud_log[-8:]:  # Last 8 entries
                text = self.font_small.render(entry, True, (255, 150, 150))
                self.screen.blit(text, (stats_x, stats_y + y_offset))
                y_offset += int(line_height * 0.8)
        
        y_offset += line_height // 2
        header2 = self.font_small.render("MANUAL REVIEW:", True, (255, 200, 0))
        self.screen.blit(header2, (stats_x, stats_y + y_offset))
        y_offset += line_height
        
        # Show recent manual reviews
        if hasattr(self, 'review_log'):
            for entry in self.review_log[-6:]:  # Last 6 entries
                text = self.font_small.render(entry, True, (255, 255, 150))
                self.screen.blit(text, (stats_x, stats_y + y_offset))
                y_offset += int(line_height * 0.8)
    
    def _handle_events(self) -> None:
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._should_quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._should_quit = True
                elif event.key == pygame.K_F3:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_m:
                    self.enable_sound = not self.enable_sound
            elif event.type == pygame.VIDEORESIZE:
                self.window_width, self.window_height = event.size
                self._calculate_layout()
    
    def render(self) -> bool:
        """
        Render one frame.
        
        Returns:
            True if user requested quit, False otherwise
        """
        if self.screen is None or self.clock is None:
            return False
        
        self._handle_events()
        if self._should_quit:
            return True
        
        # Update time
        dt = self.clock.tick(60) / 1000.0  # Convert to seconds
        self.time += dt
        
        # Spawn orb for every new transaction (1:1 ratio)
        if hasattr(self.env, "current_transaction") and self.env.current_transaction:
            txn = self.env.current_transaction
            current_step = getattr(self.env, "current_step", 0)
            
            # Check if this is a new transaction (step changed)
            if not hasattr(self, "last_step") or current_step != self.last_step:
                self._spawn_orb(
                    bank_id=txn.get("bank_id", 0),
                    is_fraud=txn.get("is_fraud", False),
                    amount=txn.get("amount", 100),
                    risk_score=txn.get("risk_score", 0.5)
                )
                self.last_step = current_step
        
        # Update systems (normal speed for clear demo)
        self._update_orbs(dt)
        self._update_particles(dt)
        self._update_timeline(dt)
        self._update_threat_meter()
        self._update_brain(dt)
        
        # Draw everything
        self._draw_background()
        self._draw_banks()
        self._draw_gates()
        self._draw_orbs()
        self._draw_particles()
        self._draw_agent_brain()
        self._draw_probability_bars()
        self._draw_threat_meter()
        self._draw_timeline()
        self._draw_stats()
        
        if self.show_debug:
            debug_text = self.font_small.render(f"FPS: {int(self.clock.get_fps())}", True, self.COLOR_TEXT_BRIGHT)
            self.screen.blit(debug_text, (10, 10))
        
        pygame.display.flip()
        return False
    
    def close(self) -> None:
        """Clean up resources"""
        pygame.quit()

