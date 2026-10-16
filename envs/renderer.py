"""
renderer.py — Pygame Renderer for AlphaeWorldEnv v2
====================================================
Clean, professional renderer for the CA grid world.
Prioritises clarity and readability over visual effects.

Layout:
    ┌──────────────────────────┬─────────────────────┐
    │   MICROALFI WORLD v2     │  header / step info  │
    ├──────────────────────────┼─────────────────────┤
    │                          │  GLOBAL CONDITIONS   │
    │     10×10 CA GRID        │  8 condition bars    │
    │                          │                      │
    │  Green  = Type A biomass │  COLONY STATS        │
    │  Cyan   = Type B mutant  │  coverage, toxin etc │
    │  Dark   = dead cell      │                      │
    │                          │  EPISODE REWARD      │
    └──────────────────────────┴─────────────────────┘

Colour scheme (simple, meaningful):
    Dead cell        — near-black      : nothing alive
    Type A low       — dark green      : alive, struggling
    Type A high      — bright green    : alive, thriving
    Type B           — blue-cyan       : mutant, visually distinct
    Toxic cell       — dark red tint   : warning overlay only

Usage:
    renderer = AlphaeRenderer(cell_size=50, fps=10)
    renderer.render(env)
    renderer.close()
"""

import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from envs.alphae_world_env_v2 import (
    AlphaeWorldEnvV2,
    GRID_H, GRID_W, TYPE_B,
    LIGHT_SWEET_LOW, LIGHT_SWEET_HIGH,
    CO2_SWEET_LOW, CO2_SWEET_HIGH,
    PH_OPTIMAL, TOXIN_SLOW_THRESH, TOXIN_KILL_THRESH,
    PHOTOINHIBITION_LIGHT, PHOTOINHIBITION_CO2,
)


# ── Colour palette ────────────────────────────────────────────────────────────
# Keep it minimal: background, text, green (alive), cyan (Type B), red (danger)

BG_MAIN      = (15,  18,  24)    # main window background
BG_PANEL     = (22,  26,  34)    # sidebar and card backgrounds
BG_HEADER    = (18,  22,  30)    # header strip

LINE_COLOR   = (50,  58,  72)    # dividers and borders

TEXT_BRIGHT  = (230, 235, 242)   # primary text, values
TEXT_DIM     = (130, 140, 158)   # labels, secondary info

# Cell colours — only two hues, green and cyan
DEAD         = (18,  22,  30)    # empty/dead cell
A_LOW        = (20,  65,  30)    # Type A, low biomass
A_MED        = (35,  130, 55)    # Type A, medium biomass
A_HIGH       = (55,  200, 80)    # Type A, high biomass
B_LOW        = (20,  80,  120)   # Type B, low biomass
B_HIGH       = (60,  180, 220)   # Type B, high biomass

# Condition bars
BAR_BG       = (35,  42,  55)    # unfilled bar track
BAR_IN       = (55,  200, 80)    # bar fill when in sweet spot (green)
BAR_OUT      = (180, 60,  60)    # bar fill when out of sweet spot (red)
BAR_NONE     = (80,  110, 160)   # bar fill when no sweet spot applies (blue-grey)

# Pulse / pulse border
PULSE_COLOR  = (220, 200, 80)    # amber, used only for pulse indicator


class AlphaeRenderer:
    """
    Pygame renderer for AlphaeWorldEnvV2.
    Call render(env) every step. Manages its own pygame lifecycle.
    """

    def __init__(self, cell_size: int = 50, fps: int = 10):
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame not installed. Run: pip install pygame")

        self.cell_size = cell_size
        self.fps       = fps

        # Layout
        self.grid_px_w = GRID_W * cell_size
        self.grid_px_h = GRID_H * cell_size
        self.pad       = 14
        self.header_h  = 36
        self.sidebar_w = 260

        self.win_w = self.grid_px_w + self.sidebar_w + self.pad * 3
        self.win_h = self.header_h + self.grid_px_h + self.pad * 2

        self.screen  = None
        self.clock   = None
        self.fonts   = {}
        self._ready  = False

        self.reward_history = []

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────

    def render(self, env: AlphaeWorldEnvV2):
        """Update the pygame window with the current environment state."""
        if not self._ready:
            self._init()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.reward_history.append(env.episode_reward)
        if len(self.reward_history) > 200:
            self.reward_history.pop(0)

        self.screen.fill(BG_MAIN)
        self._draw_header(env)
        self._draw_grid(env)
        self._draw_sidebar(env)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        """Shut down pygame."""
        if self._ready:
            pygame.quit()
            self._ready = False

    # ────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ────────────────────────────────────────────────────────────────────────

    def _init(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("Microalfi World v2 — CA Grid Simulation")
        self.clock  = pygame.time.Clock()

        # Three font sizes, all same face for consistency
        def F(size, bold=False):
            try:
                return pygame.font.SysFont("segoeui", size, bold=bold)
            except Exception:
                return pygame.font.SysFont(None, size)

        self.fonts = {
            "title":  F(16, bold=True),
            "label":  F(14),
            "value":  F(14, bold=True),
            "small":  F(13),
        }
        self._ready = True

    # ────────────────────────────────────────────────────────────────────────
    # Header
    # ────────────────────────────────────────────────────────────────────────

    def _draw_header(self, env: AlphaeWorldEnvV2):
        """Top strip: title, step counter, episode reward."""
        pygame.draw.rect(self.screen, BG_HEADER,
                         (0, 0, self.win_w, self.header_h))
        pygame.draw.line(self.screen, LINE_COLOR,
                         (0, self.header_h), (self.win_w, self.header_h))

        # Title left
        self._text("MICROALFI WORLD  v2", self.pad, 10,
                   "title", TEXT_BRIGHT)

        # Step centre
        step_str = f"Step  {env.step_count} / {env.max_steps}"
        self._text(step_str, self.win_w // 2 - 50, 10,
                   "label", TEXT_BRIGHT)

        # Reward right
        r = env.episode_reward
        col = BAR_IN if r >= 0 else BAR_OUT
        self._text(f"Reward  {r:+.1f}",
                   self.win_w - 140, 10, "value", col)

    # ────────────────────────────────────────────────────────────────────────
    # Grid
    # ────────────────────────────────────────────────────────────────────────

    def _draw_grid(self, env: AlphaeWorldEnvV2):
        """Draw the 10×10 CA colony grid."""
        ox = self.pad
        oy = self.header_h + self.pad
        cs = self.cell_size

        for i in range(GRID_H):
            for j in range(GRID_W):
                bio   = float(env.biomass[i, j])
                tox   = float(env.local_toxin[i, j])
                ctype = int(env.cell_type[i, j])

                x = ox + j * cs
                y = oy + i * cs
                rect = (x + 1, y + 1, cs - 2, cs - 2)

                # Cell fill colour
                colour = self._cell_colour(bio, tox, ctype)
                pygame.draw.rect(self.screen, colour, rect)

                # Toxic overlay: subtle red tint, not full recolour
                if bio > 0.05 and tox > TOXIN_SLOW_THRESH:
                    surf = pygame.Surface((cs - 2, cs - 2), pygame.SRCALPHA)
                    alpha = int(min(1.0, (tox - TOXIN_SLOW_THRESH) / 0.3) * 80)
                    surf.fill((180, 30, 30, alpha))
                    self.screen.blit(surf, (x + 1, y + 1))

                # Type B marker: small white square in top-right corner
                if ctype == TYPE_B and bio > 0.05:
                    pygame.draw.rect(self.screen, TEXT_BRIGHT,
                                     (x + cs - 8, y + 3, 5, 5))

        # Grid border
        pygame.draw.rect(self.screen, LINE_COLOR,
                         (ox, oy, self.grid_px_w, self.grid_px_h), 1)

        # Pulse border — only visual indicator for pulse, amber outline
        if env.pulse_steps_remaining > 0:
            pygame.draw.rect(self.screen, PULSE_COLOR,
                             (ox - 2, oy - 2,
                              self.grid_px_w + 4, self.grid_px_h + 4), 2)
            self._text(f"PULSE  {env.pulse_steps_remaining}",
                       ox + 4, oy + self.grid_px_h + 4,
                       "small", PULSE_COLOR)

        # Legend below grid
        self._draw_grid_legend(ox, oy + self.grid_px_h + 18)

    def _draw_grid_legend(self, x: int, y: int):
        """Small inline legend under the grid."""
        items = [
            (A_HIGH, "Type A — high biomass"),
            (A_LOW,  "Type A — low biomass"),
            (B_HIGH, "Type B — mutant"),
            (DEAD,   "Dead / empty"),
        ]
        cx = x
        for colour, label in items:
            pygame.draw.rect(self.screen, colour, (cx, y, 12, 12))
            pygame.draw.rect(self.screen, LINE_COLOR, (cx, y, 12, 12), 1)
            self._text(label, cx + 16, y, "small", TEXT_DIM)
            cx += 145

    # ────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ────────────────────────────────────────────────────────────────────────

    def _draw_sidebar(self, env: AlphaeWorldEnvV2):
        """Right sidebar: condition bars, colony stats, reward chart."""
        g  = env.global_state
        sx = self.grid_px_w + self.pad * 2
        sy = self.header_h + self.pad
        sw = self.sidebar_w - self.pad

        # Sidebar background
        pygame.draw.rect(self.screen, BG_PANEL,
                         (sx - self.pad // 2, sy - self.pad // 2,
                          sw + self.pad, self.win_h))

        y = sy

        # ── Global Conditions ────────────────────────────────────────────────
        y = self._section("GLOBAL CONDITIONS", sx, y, sw)

        bars = [
            ("Light",    g[0], LIGHT_SWEET_LOW,   LIGHT_SWEET_HIGH),
            ("Temp",     g[1], None,               None),
            ("CO2",      g[2], CO2_SWEET_LOW,      CO2_SWEET_HIGH),
            ("Nutrients",g[3], 0.25,               1.0),
            ("pH",       g[4], PH_OPTIMAL - 0.15,  PH_OPTIMAL + 0.15),
            ("Toxin",    g[5], None,               None),
            ("Radiation",g[6], None,               None),
            ("Humidity", g[7], 0.35,               0.75),
        ]
        for name, val, lo, hi in bars:
            y = self._bar(name, val, lo, hi, sx, y, sw)

        y += 16

        # ── Colony Stats ─────────────────────────────────────────────────────
        y = self._section("COLONY STATS", sx, y, sw)

        living     = env.biomass > 0.05
        n_living   = max(1, int(living.sum()))
        coverage   = living.mean() * 100
        total_bio  = float(env.biomass.sum())
        n_type_b   = int(np.sum((env.cell_type == TYPE_B) & living))
        mean_tox   = float(env.local_toxin[living].mean()) if n_living > 0 else 0.0

        stats = [
            ("Coverage",   f"{coverage:.1f}%",
             BAR_IN if coverage > 50 else BAR_OUT),
            ("Total biomass", f"{total_bio:.2f}",  TEXT_BRIGHT),
            ("Living cells",  f"{n_living}",        TEXT_BRIGHT),
            ("Type B cells",  f"{n_type_b}",
             B_HIGH if n_type_b > 0 else TEXT_DIM),
            ("Mean toxin",    f"{mean_tox:.3f}",
             BAR_OUT if mean_tox > TOXIN_SLOW_THRESH else TEXT_BRIGHT),
        ]

        for label, value, col in stats:
            self._text(label, sx, y, "label", TEXT_DIM)
            self._text(value, sx + sw - self._text_w(value, "value"), y,
                       "value", col)
            y += 22

        y += 14

        # ── Reward Chart ─────────────────────────────────────────────────────
        y = self._section("EPISODE REWARD", sx, y, sw)
        r_col = BAR_IN if env.episode_reward >= 0 else BAR_OUT
        self._text(f"{env.episode_reward:+.2f}", sx, y, "value", r_col)
        y += 24

        if len(self.reward_history) >= 2:
            self._sparkline(sx, y, sw, 90)

    # ────────────────────────────────────────────────────────────────────────
    # Drawing primitives
    # ────────────────────────────────────────────────────────────────────────

    def _section(self, title: str, x: int, y: int, w: int) -> int:
        """Section header with underline. Returns new y."""
        self._text(title, x, y, "title", TEXT_BRIGHT)
        y += 20
        pygame.draw.line(self.screen, LINE_COLOR, (x, y), (x + w, y))
        return y + 8

    def _bar(self, name: str, val: float,
             lo, hi, x: int, y: int, w: int) -> int:
        """
        Condition bar with label and value.
        Green when inside sweet spot, red when outside, blue-grey when no sweet spot.
        """
        bar_h  = 11
        lbl_w  = 88
        val_w  = 38
        bar_w  = w - lbl_w - val_w - 8

        # Label
        self._text(f"{name}", x, y, "small", TEXT_DIM)

        bx = x + lbl_w
        by = y

        # Track
        pygame.draw.rect(self.screen, BAR_BG, (bx, by, bar_w, bar_h))

        # Sweet spot zone marker (thin darker strip)
        if lo is not None and hi is not None:
            zx1 = bx + int(lo * bar_w)
            zx2 = bx + int(hi * bar_w)
            pygame.draw.rect(self.screen, (50, 80, 50),
                             (zx1, by, zx2 - zx1, bar_h))

        # Bar fill
        fill = int(np.clip(val, 0, 1) * bar_w)
        if lo is not None and hi is not None:
            colour = BAR_IN if lo <= val <= hi else BAR_OUT
        else:
            colour = BAR_NONE
        if fill > 0:
            pygame.draw.rect(self.screen, colour, (bx, by, fill, bar_h))

        # Border
        pygame.draw.rect(self.screen, LINE_COLOR, (bx, by, bar_w, bar_h), 1)

        # Value
        self._text(f"{val:.2f}", bx + bar_w + 4, by, "small", TEXT_DIM)

        return y + bar_h + 8

    def _sparkline(self, x: int, y: int, w: int, h: int):
        """Mini reward history line chart."""
        rewards = self.reward_history
        if len(rewards) < 2:
            return

        mn, mx = min(rewards), max(rewards)
        rng    = max(mx - mn, 1.0)

        pygame.draw.rect(self.screen, BG_MAIN, (x, y, w, h))
        pygame.draw.rect(self.screen, LINE_COLOR, (x, y, w, h), 1)

        # Zero line
        zy = y + int(h * (1 - (0 - mn) / rng))
        zy = int(np.clip(zy, y, y + h))
        pygame.draw.line(self.screen, LINE_COLOR, (x, zy), (x + w, zy))

        # Line
        pts = []
        for idx, r in enumerate(rewards):
            px = x + int(idx / (len(rewards) - 1) * (w - 1))
            py = y + h - int((r - mn) / rng * (h - 2)) - 1
            py = int(np.clip(py, y + 1, y + h - 1))
            pts.append((px, py))

        for k in range(len(pts) - 1):
            col = BAR_IN if rewards[k] >= 0 else BAR_OUT
            pygame.draw.line(self.screen, col, pts[k], pts[k + 1], 1)

    def _text(self, msg: str, x: int, y: int, size: str, colour):
        """Render text at position."""
        surf = self.fonts[size].render(str(msg), True, colour)
        self.screen.blit(surf, (x, y))

    def _text_w(self, msg: str, size: str) -> int:
        """Return pixel width of a text string."""
        return self.fonts[size].size(str(msg))[0]

    # ────────────────────────────────────────────────────────────────────────
    # Cell colour
    # ────────────────────────────────────────────────────────────────────────

    def _cell_colour(self, bio: float, tox: float, ctype: int):
        """
        Cell colour based on biomass level and type.
        Type A: green gradient. Type B: cyan gradient. Dead: near-black.
        """
        if bio <= 0.05:
            return DEAD

        t = min(1.0, bio / 2.0)   # normalise biomass to [0, 1] for gradient

        if ctype == TYPE_B:
            return self._lerp(B_LOW, B_HIGH, t)

        # Type A: green gradient
        if bio < 0.5:
            return self._lerp(A_LOW, A_MED, bio / 0.5)
        else:
            return self._lerp(A_MED, A_HIGH, min(1.0, (bio - 0.5) / 1.5))

    @staticmethod
    def _lerp(c1, c2, t):
        t = float(np.clip(t, 0.0, 1.0))
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )