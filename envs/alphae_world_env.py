"""
AlphaeWorldEnv — v1 (Deterministic Core)
========================================
A Gymnasium environment inspired by microalphae cultivation,
but set in an alien world with its own internal rules.

This is NOT a real biology simulation. The rules are internally
consistent and interesting enough for RL agents to learn, but
they are not bound by real-world microalphae behaviour.

State space (6 continuous dimensions):
  [0] light       — 0.0 to 1.0  (intensity fraction)
  [1] temperature — 0.0 to 1.0  (normalised alien scale)
  [2] co2         — 0.0 to 1.0  (concentration fraction)
  [3] nutrients   — 0.0 to 1.0  (remaining fraction)
  [4] ph          — 0.0 to 1.0  (0 = acid, 1 = alkaline)
  [5] biomass     — 0.0 to 5.0  (g/L equivalent, grows over time)

Action space (discrete, 7 actions):
  0 — do nothing          (cost: 0.00)
  1 — increase light      (cost: 0.05)
  2 — decrease light      (cost: 0.01)
  3 — increase co2        (cost: 0.04)
  4 — decrease co2        (cost: 0.01)
  5 — add nutrients       (cost: 0.06)
  6 — adjust pH (buffer)  (cost: 0.03)

Reward per step:
  r = (biomass_gained * 10) - action_cost - pH_penalty

Alien world rules:
  - Light sweet spot  : 0.55–0.75  → up to 1.4x growth multiplier
  - CO2 sweet spot    : 0.40–0.65  → up to 1.3x growth multiplier
  - Photoinhibition   : light > 0.85 AND co2 > 0.80 → biomass damage
  - Temperature       : drifts toward 0.30 each step (cold snap)
  - pH                : drops as biomass grows (acidification)
  - Nutrients         : depleted proportional to growth rate

Episode ends when:
  - biomass <= 0.0   (culture crash)
  - step >= max_steps (default 200)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── World constants ──────────────────────────────────────────────────────────
# These define the "physics" of our alien world. Changing these values
# changes the challenge difficulty — e.g. narrowing the sweet spot makes
# the problem harder for RL agents to solve.

# Sweet spots: the optimal range for each variable.
# Growth is maximised when both light AND co2 are inside their sweet spots.
# Deliberately offset from each other so the agent must find two separate targets.
LIGHT_SWEET_LOW   = 0.55
LIGHT_SWEET_HIGH  = 0.75
CO2_SWEET_LOW     = 0.40
CO2_SWEET_HIGH    = 0.65

# The pH value that gives maximum growth bonus.
# Below this, acidification from growth will start hurting the culture.
PH_OPTIMAL        = 0.60

# Temperature drifts passively toward 0.30 (the alien "cold snap").
# The agent has no action to counter this — it is background pressure
# that makes the environment non-stationary over the episode.
TEMP_DRIFT_TARGET = 0.30
TEMP_DRIFT_RATE   = 0.005   # fraction of remaining gap closed per step

# Base growth rate: how much biomass is added per step under neutral conditions
# (multipliers of 1.0 across all variables). In practice growth is usually
# higher when conditions are good and lower when they are not.
BASE_GROWTH_RATE  = 0.02

# How fast pH drops per unit of biomass grown.
# Faster growth → more acidification → more pH penalty if not buffered.
PH_ACIDIFICATION  = 0.003

# How fast nutrients are consumed per unit of biomass grown.
# If nutrients hit 0, growth stops entirely (hard wall).
NUTRIENT_UPTAKE   = 0.015

# Photoinhibition thresholds: if BOTH light AND co2 exceed these at the same
# time, the organism takes damage. This is a deliberate trap that punishes
# the naive strategy of "max everything out".
PHOTOINHIBITION_LIGHT  = 0.85
PHOTOINHIBITION_CO2    = 0.80
PHOTOINHIBITION_DAMAGE = -0.04   # biomass removed per step while in this zone

# Action costs: how much each action subtracts from the reward.
# Increasing resources (light, co2, nutrients) is expensive because it
# consumes energy or physical inputs. Decreasing is cheap — just venting
# or dimming. Doing nothing is free, which encourages the agent to wait
# rather than constantly intervening.
ACTION_COSTS = {
    0: 0.00,   # do nothing       — always free
    1: 0.05,   # increase light   — highest energy cost
    2: 0.01,   # decrease light   — cheap, just dim
    3: 0.04,   # increase co2     — requires gas injection
    4: 0.01,   # decrease co2     — cheap, just vent
    5: 0.06,   # add nutrients    — most expensive (physical resource)
    6: 0.03,   # adjust pH        — buffer chemical, moderate cost
}

# How much each action shifts the target variable.
# 0.08 is a deliberate choice: small enough that the agent must take
# multiple steps to reach the sweet spot from a bad starting position,
# large enough that recovery is feasible within 200 steps.
ACTION_DELTA = 0.08


class AlphaeWorldEnv(gym.Env):
    """
    Gymnasium environment for alien microalphae cultivation.

    Fully compatible with Gymnasium's standard API and Stable-Baselines3.
    The agent observes a 6-dimensional state vector each step, picks one of
    7 discrete actions, and receives a scalar reward based on biomass growth
    minus the cost of the chosen action.
    """

    # render_modes tells Gymnasium what rendering options are available.
    # "human" prints to stdout, "ansi" returns a string, "rgb_array" is
    # a placeholder for future pixel-based rendering.
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, max_steps: int = 200, render_mode=None):
        super().__init__()
        self.max_steps   = max_steps
        self.render_mode = render_mode

        # ── Observation space ────────────────────────────────────────────────
        # A Box space means a continuous vector with per-element min/max bounds.
        # Biomass goes up to 5.0 (not 1.0) because it grows over the episode.
        # All other variables are normalised to [0, 1].
        low  = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # ── Action space ─────────────────────────────────────────────────────
        # Discrete(7) means integer actions 0, 1, 2, ..., 6.
        # We use discrete rather than continuous actions so DQN (which requires
        # discrete actions) can be used alongside PPO and SAC.
        self.action_space = spaces.Discrete(7)

        # ── Internal state (populated by reset()) ────────────────────────────
        self.state          = None   # numpy array of shape (6,)
        self.step_count     = 0
        self.episode_reward = 0.0
        self.history        = []     # list of state dicts, one per step

    # ────────────────────────────────────────────────────────────────────────
    # Gymnasium API — these three methods are the required interface
    # ────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Reset the environment to a fresh starting state.

        Called at the beginning of each episode. Starting conditions are
        randomised within reasonable ranges so the agent learns to generalise
        across different initial states, not just memorise one fixed scenario.

        Returns:
            obs  : initial observation (numpy array, shape (6,))
            info : diagnostic dict (in_light_sweet, in_co2_sweet, etc.)
        """
        # Must call super().reset(seed) to properly seed self.np_random,
        # which is the Gymnasium-managed RNG used for reproducibility.
        super().reset(seed=seed)
        rng = self.np_random

        self.state = np.array([
            rng.uniform(0.3, 0.7),   # light    — somewhere in the middle range
            rng.uniform(0.4, 0.7),   # temp     — will drift downward from here
            rng.uniform(0.3, 0.6),   # co2      — may need adjusting immediately
            rng.uniform(0.7, 1.0),   # nutrients — start fairly full
            rng.uniform(0.5, 0.7),   # pH       — starts near optimal, will drop
            rng.uniform(0.1, 0.3),   # biomass  — small starting inoculum
        ], dtype=np.float32)

        # Reset episode tracking
        self.step_count     = 0
        self.episode_reward = 0.0
        self.history        = [self._state_dict()]   # record initial state

        return self.state.copy(), self._info()

    def step(self, action: int):
        """
        Apply one action and advance the world by one timestep.

        This is the core loop. Order of operations matters:
          1. Apply the action's effect on the state
          2. Run world dynamics (growth, consumption, drift)
          3. Clip everything to valid bounds
          4. Compute reward from the resulting state change
          5. Check termination conditions

        Returns:
            obs        : new observation after the step (shape (6,))
            reward     : scalar reward for this step
            terminated : True if the culture crashed (biomass hit 0)
            truncated  : True if we hit the step limit (max_steps)
            info       : diagnostic dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Record biomass before the step so we can compute the delta for reward
        prev_biomass = float(self.state[5])

        # Step 1: apply the chosen action (mutates self.state in-place)
        self._apply_action(action)

        # Step 2: run the alien world dynamics (growth, consumption, drift)
        self._world_step()

        # Step 3: hard-clip state to valid bounds.
        # This is a safety net — dynamics should stay in bounds, but floating
        # point edge cases can occasionally push values slightly out of range.
        self.state = np.clip(
            self.state,
            0.0,
            np.array([1, 1, 1, 1, 1, 5], dtype=np.float32)
        )

        # Step 4: compute reward
        # biomass_delta * 10  — scaled up so growth signal dominates noise
        # action_cost         — discourages unnecessary interventions
        # ph_penalty          — continuous penalty for letting pH drift from optimal
        biomass_delta = float(self.state[5]) - prev_biomass
        action_cost   = ACTION_COSTS[action]
        ph_penalty    = abs(self.state[4] - PH_OPTIMAL) * 0.5
        reward        = biomass_delta * 10.0 - action_cost - ph_penalty

        self.step_count     += 1
        self.episode_reward += reward
        self.history.append(self._state_dict(action=action, reward=reward))

        # Step 5: check termination
        # terminated = natural end (culture crash) — the agent caused this
        # truncated  = artificial end (time limit)  — the episode just ran out
        terminated = bool(self.state[5] <= 0.0)
        truncated  = bool(self.step_count >= self.max_steps)

        info = self._info()

        # Add episode summary to info on final step (SB3 uses this for logging)
        if terminated or truncated:
            info["episode"] = {
                "r": round(self.episode_reward, 3),
                "l": self.step_count,
                "final_biomass": round(float(self.state[5]), 4),
            }

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        """Render current state. 'human' prints to stdout, 'ansi' returns string."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        print(self._render_ansi())

    def close(self):
        """Clean up resources. Nothing to close for this env, but required by API."""
        pass

    # ────────────────────────────────────────────────────────────────────────
    # World mechanics — the "physics engine" of the alien world
    # ────────────────────────────────────────────────────────────────────────

    def _apply_action(self, action: int):
        """
        Mutate self.state based on the chosen action.

        Each action shifts the relevant variable by ACTION_DELTA (0.08),
        clamped so it cannot go out of [0, 1]. Nutrients refill by a larger
        amount (0.15) because they are a bulk resource, not a fine control.
        """
        s = self.state
        d = ACTION_DELTA

        if   action == 0: pass                           # do nothing
        elif action == 1: s[0] = min(1.0, s[0] + d)     # increase light
        elif action == 2: s[0] = max(0.0, s[0] - d)     # decrease light
        elif action == 3: s[2] = min(1.0, s[2] + d)     # increase co2
        elif action == 4: s[2] = max(0.0, s[2] - d)     # decrease co2
        elif action == 5: s[3] = min(1.0, s[3] + 0.15)  # add nutrients (bulk)
        elif action == 6: s[4] = min(1.0, s[4] + d)     # buffer pH upward

    def _world_step(self):
        """
        Apply all passive alien world dynamics for one timestep.

        This runs AFTER the action has been applied. The order of operations
        within this function also matters — photoinhibition is checked first
        because if it triggers, we skip normal growth entirely.
        """
        s = self.state
        light, temp, co2, nutrients, ph, biomass = s

        # ── Photoinhibition: alien photodamage ───────────────────────────────
        # If BOTH light and co2 are simultaneously too high, the organism is
        # damaged. This is the key "trap" in the environment — naive agents
        # that just maximise all resources will stumble into this and lose
        # biomass until they learn to avoid it.
        if light > PHOTOINHIBITION_LIGHT and co2 > PHOTOINHIBITION_CO2:
            s[5] = max(0.0, biomass + PHOTOINHIBITION_DAMAGE)
            s[2] = max(0.0, co2 - 0.01)     # co2 still depletes slightly
            self._drift_temperature()         # temperature still drifts
            return                            # skip normal growth this step

        # ── Growth multipliers ───────────────────────────────────────────────
        # Each variable contributes a multiplier to the base growth rate.
        # The bell curve function returns values from ~0.05 (far outside sweet
        # spot) up to the specified peak (at the centre of the sweet spot).
        # All four multipliers are then multiplied together, so poor conditions
        # in any one variable drag down the entire growth rate.

        # Light: most important driver (peak 1.4x)
        light_mult = self._bell_curve(light, LIGHT_SWEET_LOW, LIGHT_SWEET_HIGH, peak=1.4)

        # CO2: secondary driver (peak 1.3x)
        co2_mult   = self._bell_curve(co2, CO2_SWEET_LOW, CO2_SWEET_HIGH, peak=1.3)

        # pH: tertiary modifier (peak 1.2x, narrower window of ±0.15 around optimal)
        ph_mult    = self._bell_curve(ph, PH_OPTIMAL - 0.15, PH_OPTIMAL + 0.15, peak=1.2)

        # Nutrients: square-root saturation model.
        # At full nutrients (1.0): multiplier = 1.0.
        # At half nutrients (0.5): multiplier = 0.71 — diminishing returns.
        # At near-zero (0.01):     multiplier ≈ 0.1  — almost no growth.
        # This avoids a hard cliff while still making nutrient management important.
        nutrient_mult = nutrients ** 0.5

        # ── Compute gross growth rate ────────────────────────────────────────
        # All four multipliers combine multiplicatively. This means the growth
        # rate degrades gracefully as conditions worsen rather than switching
        # off abruptly — which gives the RL agent a cleaner gradient to follow.
        growth = BASE_GROWTH_RATE * light_mult * co2_mult * ph_mult * nutrient_mult

        # ── Update state variables from growth ───────────────────────────────

        # Biomass increases by the growth amount (capped at 5.0)
        s[5] = min(5.0, biomass + growth)

        # Nutrients are consumed proportional to growth rate.
        # Scaled relative to BASE_GROWTH_RATE so consumption stays consistent
        # regardless of what BASE_GROWTH_RATE is set to.
        s[3] = max(0.0, nutrients - growth * NUTRIENT_UPTAKE / BASE_GROWTH_RATE)

        # pH drops as the organism grows (metabolic acidification).
        # Faster growth = more acidification = more pH penalty if not buffered.
        s[4] = max(0.0, ph - growth * PH_ACIDIFICATION / BASE_GROWTH_RATE)

        # CO2 is drawn down by photosynthetic activity.
        # The 0.5 factor means CO2 depletes at half the rate of growth —
        # if the agent never adds CO2, it will eventually drift out of the
        # sweet spot as the organism consumes it.
        s[2] = max(0.0, co2 - growth * 0.5)

        # ── Passive temperature drift ────────────────────────────────────────
        self._drift_temperature()

    def _drift_temperature(self):
        """
        Passively move temperature toward the cold snap target each step.

        Uses exponential smoothing: each step, close 0.5% of the remaining
        gap between current temp and target. This means temperature never
        actually reaches 0.30 — it asymptotically approaches it. This is
        intentional: it keeps the environment non-stationary throughout the
        episode without ever hitting a hard boundary.
        """
        self.state[1] += (TEMP_DRIFT_TARGET - self.state[1]) * TEMP_DRIFT_RATE

    @staticmethod
    def _bell_curve(x: float, lo: float, hi: float, peak: float = 1.0) -> float:
        """
        Smooth growth multiplier function centred on the sweet spot [lo, hi].

        Returns `peak` at the exact midpoint of [lo, hi], smoothly decaying
        to 0.05 beyond twice the half-width of the interval.

        Why a cosine taper instead of a Gaussian or linear ramp?
          - Smoother than linear (no sharp kinks at the boundary)
          - Bounded (unlike Gaussian which has infinite tails)
          - Simple to compute, easy to reason about

        Args:
            x    : current value of the variable (e.g. light = 0.62)
            lo   : lower bound of sweet spot (e.g. 0.55)
            hi   : upper bound of sweet spot (e.g. 0.75)
            peak : maximum multiplier at the centre of the sweet spot

        Returns:
            float multiplier in range [0.05, peak]

        Example:
            _bell_curve(0.65, 0.55, 0.75, peak=1.4)  →  1.4   (dead centre)
            _bell_curve(0.55, 0.55, 0.75, peak=1.4)  →  0.95  (at edge)
            _bell_curve(0.30, 0.55, 0.75, peak=1.4)  →  0.05  (far outside)
        """
        mid   = (lo + hi) / 2.0      # centre of sweet spot
        width = (hi - lo) / 2.0      # half-width of sweet spot

        dist = abs(x - mid)          # distance from centre

        # Beyond 2× the half-width, the multiplier bottoms out at 0.05.
        # The 2× factor gives a gentle tail so the agent still gets a weak
        # signal when starting from a very bad position.
        if dist >= width * 2:
            return 0.05

        # Cosine taper: t goes from 0 (at centre) to 1 (at the 2× boundary).
        # cos(π * t) goes from 1 down to -1, so (0.5 + 0.5*cos) goes 1 → 0.
        t = dist / (width * 2)
        return peak * (0.5 + 0.5 * np.cos(np.pi * t))

    # ────────────────────────────────────────────────────────────────────────
    # Helper methods
    # ────────────────────────────────────────────────────────────────────────

    def _state_dict(self, action=None, reward=None) -> dict:
        """
        Snapshot the current state as a plain Python dict.
        Used to build self.history for logging and visualisation.
        Rounds values to 4 decimal places for readability.
        """
        s = self.state
        d = {
            "step":      self.step_count,
            "light":     round(float(s[0]), 4),
            "temp":      round(float(s[1]), 4),
            "co2":       round(float(s[2]), 4),
            "nutrients": round(float(s[3]), 4),
            "ph":        round(float(s[4]), 4),
            "biomass":   round(float(s[5]), 4),
        }
        # Only add action/reward keys if they were taken this step
        if action is not None: d["action"] = action
        if reward is not None: d["reward"] = round(reward, 4)
        return d

    def _info(self) -> dict:
        """
        Return a diagnostic info dict for the current state.

        These flags are useful for:
          - Debugging agent behaviour during development
          - Tracking how often the agent is in the sweet spot (a training metric)
          - SB3's VecEnv wrappers (which pass info through unchanged)
        """
        s = self.state
        in_light_sweet  = LIGHT_SWEET_LOW  <= s[0] <= LIGHT_SWEET_HIGH
        in_co2_sweet    = CO2_SWEET_LOW    <= s[2] <= CO2_SWEET_HIGH
        photoinhibition = s[0] > PHOTOINHIBITION_LIGHT and s[2] > PHOTOINHIBITION_CO2
        return {
            "step":             self.step_count,
            "in_light_sweet":   bool(in_light_sweet),
            "in_co2_sweet":     bool(in_co2_sweet),
            "photoinhibition":  bool(photoinhibition),
            "nutrient_warning": bool(s[3] < 0.15),   # nutrients getting critically low
        }

    def _render_ansi(self) -> str:
        """
        Render the current state as a simple ASCII bar chart.
        Each bar is 20 characters wide, normalised to its max value.
        Biomass is normalised to 5.0; all other variables to 1.0.
        """
        s = self.state
        names = ["Light", "Temp ", "CO2  ", "Nutr ", "pH   ", "Bio  "]
        bars  = []
        for i, (name, val) in enumerate(zip(names, s)):
            max_val = 5.0 if i == 5 else 1.0
            pct     = val / max_val
            filled  = int(pct * 20)
            bar     = "█" * filled + "░" * (20 - filled)
            bars.append(f"  {name}: [{bar}] {val:.3f}")
        lines = [
            f"\n── AlphaeWorld Step {self.step_count} ──",
            *bars,
            f"  Ep.reward: {self.episode_reward:.3f}",
        ]
        return "\n".join(lines)