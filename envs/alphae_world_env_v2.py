"""
AlphaeWorldEnv v2 -- CA Grid World
====================================
A 10x10 cellular automaton grid environment where each cell is an
independent agent governed by local rules. A single RL agent sits
OUTSIDE the grid and controls global conditions. Global behaviour
(colony spread, mutation, toxin buildup) emerges from local interactions.

Architecture:
    RL Agent (outside)
        -- observes: global conditions + grid statistics
        -- acts on:  global parameters (light, CO2, etc.)
        -- reward:   configurable via reward_mode parameter
            |
            v
    Global Conditions (8 variables, some drifting passively)
        light, temperature, CO2, nutrients, pH,
        toxin_level, radiation, humidity
            |
            v  influences every cell
    10x10 CA Grid
        Each cell tracks: biomass, age, local_toxin, cell_type
        Each cell follows LOCAL rules only -- emergence happens here

Global state space (8 continuous dimensions):
    [0] light       -- 0.0 to 1.0
    [1] temperature -- 0.0 to 1.0  (drifts toward cold snap)
    [2] co2         -- 0.0 to 1.0  (drawn down by colony + natural drift)
    [3] nutrients   -- 0.0 to 1.0  (consumed by colony, degradation rate increases)
    [4] ph          -- 0.0 to 1.0  (acidified by growth + environmental drift)
    [5] toxin       -- 0.0 to 1.0  (byproduct of growth)
    [6] radiation   -- 0.0 to 1.0  (fluctuates randomly each step)
    [7] humidity    -- 0.0 to 1.0  (affects CA spreading rate)

RL agent observation (8 global + 5 grid stats = 13 dimensions):
    [0-7]  global conditions
    [8]    mean_biomass     -- average biomass across all 100 cells
    [9]    colony_coverage  -- fraction of cells with biomass > 0.05 (0 to 1)
    [10]   mean_toxin       -- average local toxin across living cells (0 to 1)
    [11]   type_b_fraction  -- fraction of living cells that are Type B (0 to 1)
    [12]   mean_age         -- average age of living cells, normalised (0 to 1)

Action space (discrete, 11 actions):
    0  -- do nothing              (cost: 0.00)
    1  -- increase light          (cost: 0.05)
    2  -- decrease light          (cost: 0.01)
    3  -- increase co2            (cost: 0.04)
    4  -- decrease co2            (cost: 0.01)
    5  -- add nutrients           (cost: 0.06)
    6  -- adjust pH               (cost: 0.03)
    7  -- activate toxin filter   (cost: 0.07)
    8  -- increase radiation shield (cost: 0.06)
    9  -- add humidity            (cost: 0.03)
    10 -- trigger growth pulse    (cost: 0.08)

Reward modes (set via reward_mode parameter at init):
    "multi_objective"  -- biomass + coverage - toxin - cost - pH  [default]
    "biomass_only"     -- biomass delta scaled, minus cost only
    "turbidity"        -- colony coverage as proxy for yield (like real turbidity)
    "turbidity_delta"  -- change in colony coverage since previous step, minus cost
    "efficiency"       -- biomass gained per unit of total action cost
    "sustainability"   -- like multi_objective but double toxin penalty

World config (set via world_config dict at init):
    Toggles which CA rules are active, creating worlds of different difficulty.
    Default: all rules active (hardest world).
    Keys: "spreading", "mutation", "toxin", "colony_support", "photoinhibition"

Drifting global variables (passive, agent cannot stop them unless no_drifts=True):
    - Temperature:   drifts toward cold snap target
    - CO2:           natural outgassing drift downward
    - Nutrients:     degradation rate increases over episode time
    - pH:            slow environmental acidification
    - Light:         sinusoidal day/night cycle variation
    - Radiation:     random noise each step (always active even with no_drifts)
    - Humidity:      drifts toward moderate baseline

Episode ends when:
    - colony_coverage < 0.05  (near-extinction)
    - step >= max_steps        (default 200)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -- World constants ----------------------------------------------------------

LIGHT_SWEET_LOW   = 0.55
LIGHT_SWEET_HIGH  = 0.75
CO2_SWEET_LOW     = 0.40
CO2_SWEET_HIGH    = 0.65
PH_OPTIMAL        = 0.60
HUMIDITY_OPTIMAL  = 0.55

# -- Drifting global variable parameters --------------------------------------

# NOTE (intentional, kept for reproducibility): temperature is observed
# (global_state[1] / obs[1]) and drifts every step, but it has NO effect on
# growth anywhere in _ca_step, and no action controls it. It exists purely
# as a non-actionable, non-influential background signal. Do not "fix" this
# without creating a new env version -- it would change every existing
# result for every reward mode.
TEMP_DRIFT_TARGET = 0.30
TEMP_DRIFT_RATE   = 0.005

# CO2 natural outgassing independent of colony activity
CO2_NATURAL_DRIFT = 0.001

# Nutrient degradation rate increases over episode time (media aging)
NUTRIENT_DEGRADATION_BASE = 0.001
NUTRIENT_DEGRADATION_RATE = 0.00002

# Environmental pH acidification independent of colony growth
PH_ENV_DRIFT = 0.0008

# Light sinusoidal day/night cycle
LIGHT_CYCLE_AMPLITUDE = 0.10
LIGHT_CYCLE_PERIOD    = 100

# Radiation random fluctuation each step (always active)
RADIATION_NOISE  = 0.05
RADIATION_SHIELD = 0.15

# -- Growth dynamics ----------------------------------------------------------

BASE_GROWTH_RATE       = 0.02
PH_ACIDIFICATION       = 0.003
NUTRIENT_UPTAKE        = 0.015
PHOTOINHIBITION_LIGHT  = 0.85
PHOTOINHIBITION_CO2    = 0.80
PHOTOINHIBITION_DAMAGE = -0.04

# -- Toxin dynamics -----------------------------------------------------------

TOXIN_PRODUCTION_A = 0.012
TOXIN_PRODUCTION_B = 0.004
# NOTE (intentional, kept for reproducibility): TOXIN_DECAY only lowers the
# *reported global* toxin value (global_state[5] / obs[5]), which is
# recomputed from mean local toxin minus this constant every step in
# _global_dynamics. It does NOT decay self.local_toxin itself. Per-cell
# local_toxin only ever decreases via action 7 (toxin filter); otherwise it
# is monotonically non-decreasing (growth, photoinhibition and spreading all
# only add to it). Do not "fix" this asymmetry without a new env version.
TOXIN_DECAY        = 0.005
TOXIN_FILTER_BOOST = 0.06
TOXIN_SLOW_THRESH  = 0.70
TOXIN_KILL_THRESH  = 0.90

# -- CA spreading -------------------------------------------------------------

SPREAD_RATE         = 0.08
HUMIDITY_SPREAD_MOD = 0.5

# Minimum biomass a cell must have before it spreads to neighbours.
# Set to 0.10 (above the 0.05 alive threshold) so spreading cells
# cannot accidentally drop below the alive threshold and kill themselves,
# while still allowing moderately healthy cells to spread.
SPREAD_MIN_BIOMASS  = 0.10

# -- Colony support -----------------------------------------------------------

COLONY_SUPPORT_THRESH = 4
COLONY_SUPPORT_BONUS  = 1.2
ISOLATION_THRESH      = 0
ISOLATION_PENALTY     = 0.5

# -- Mutation -----------------------------------------------------------------

MUTATION_AGE_THRESH = 50
MUTATION_BIO_THRESH = 0.8
MUTATION_PROB       = 0.05

# -- Growth pulse -------------------------------------------------------------

PULSE_DURATION     = 3
PULSE_SPREAD_BOOST = 3.0

# -- Cell type encoding -------------------------------------------------------

TYPE_A = 0
TYPE_B = 1

# -- Grid ---------------------------------------------------------------------

GRID_H  = 10
GRID_W  = 10
N_CELLS = GRID_H * GRID_W

# -- Action costs -------------------------------------------------------------

ACTION_COSTS = {
    0:  0.00,
    1:  0.05,
    2:  0.01,
    3:  0.04,
    4:  0.01,
    5:  0.06,
    6:  0.03,
    7:  0.07,
    8:  0.06,
    9:  0.03,
    10: 0.08,
}

ACTION_DELTA = 0.08

# -- Reward weights -----------------------------------------------------------

W_BIOMASS  = 10.0
W_COVERAGE = 2.0
W_COST     = 1.0
W_TOXIN    = 3.0
W_PH       = 0.5

# -- Termination --------------------------------------------------------------

EXTINCTION_COVERAGE = 0.05

# -- Default world config: all rules active -----------------------------------

DEFAULT_WORLD_CONFIG = {
    "spreading":       True,
    "mutation":        True,
    "toxin":           True,
    "colony_support":  True,
    "photoinhibition": True,
}


class AlphaeWorldEnvV2(gym.Env):
    """
    CA grid world environment for Microalfi cultivation.

    10x10 grid where each cell follows local CA rules independently.
    The RL agent controls global conditions only. Colony behaviour emerges
    from cell-to-cell interactions without any central coordination.

    Args:
        max_steps    : episode length (default 200)
        render_mode  : "human" for ANSI output, None for headless
        reward_mode  : which reward function to use
        world_config : dict of booleans toggling CA rules
        no_drifts    : if True, disable all passive global drifts except
                       radiation noise (which is always active)
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(
        self,
        max_steps:    int  = 200,
        render_mode:  str  = None,
        reward_mode:  str  = "multi_objective",
        world_config: dict = None,
        no_drifts:    bool = False,
    ):
        super().__init__()
        self.max_steps   = max_steps
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.no_drifts   = no_drifts

        # Merge supplied world_config with defaults
        self.world_config = {**DEFAULT_WORLD_CONFIG, **(world_config or {})}

        # Observation space: 8 global + 5 grid stats = 13 dimensions
        low  = np.zeros(13, dtype=np.float32)
        high = np.array([1,1,1,1,1,1,1,1, 5, 1,1,1,1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 11 discrete actions
        self.action_space = spaces.Discrete(11)

        # Internal state (initialised in reset)
        self.global_state          = None
        self.biomass               = None
        self.cell_age              = None
        self.local_toxin           = None
        self.cell_type             = None
        self.step_count            = 0
        self.episode_reward        = 0.0
        self.pulse_steps_remaining = 0
        self.total_action_cost     = 0.0
        self.history               = []
        self.prev_coverage         = 0.0

    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Reset to a fresh episode. Colony seeded in centre 4x4 region.

        Starting biomass range is 0.15 to 0.35 per cell. This minimum is
        above the 0.05 alive threshold and the 0.10 spread minimum, so
        seeded cells are never dangerously close to dying or spreading
        before they have had a chance to grow (see the seeding block below).
        """
        super().reset(seed=seed)
        rng = self.np_random

        self.global_state = np.array([
            rng.uniform(0.3, 0.7),   # light
            rng.uniform(0.4, 0.7),   # temperature
            rng.uniform(0.3, 0.6),   # co2
            rng.uniform(0.7, 1.0),   # nutrients
            rng.uniform(0.5, 0.7),   # pH
            rng.uniform(0.0, 0.2),   # toxin (starts low)
            rng.uniform(0.2, 0.4),   # radiation
            rng.uniform(0.3, 0.7),   # humidity
        ], dtype=np.float32)

        self.biomass     = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.cell_age    = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.local_toxin = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        self.cell_type   = np.full((GRID_H, GRID_W), TYPE_A, dtype=np.int32)

        # Seed centre 4x4 region with biomass 0.15-0.35.
        # Range is above the 0.05 alive threshold and the 0.10 spread minimum
        # so cells can survive and spread from step 1, but low enough that
        # bad starting conditions still challenge the agent.
        for i in range(3, 7):
            for j in range(3, 7):
                if rng.random() > 0.3:   # 70% of seed cells start alive
                    self.biomass[i, j] = rng.uniform(0.15, 0.35)

        self.step_count            = 0
        self.episode_reward        = 0.0
        self.pulse_steps_remaining = 0
        self.total_action_cost     = 0.0
        self.history               = [self._state_dict()]
        self.prev_coverage         = 0.0

        return self._get_obs(), self._info()

    def step(self, action: int):
        """Apply action, run CA step, update passive dynamics, compute reward."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        prev_total_biomass = float(self.biomass.sum())

        # 1. Apply agent action
        self._apply_action(action)

        # 2. Run CA local rules
        self._ca_step()

        # 3. Passive global dynamics (skipped if no_drifts=True except radiation)
        self._global_dynamics()

        # 4. Clip all values to valid bounds.
        # Biomass is capped at 5.0 per cell -- with a 10x10 = 100-cell grid,
        # that is a hard ceiling of 500.0 on total biomass (env.biomass.sum()).
        self.global_state = np.clip(self.global_state, 0.0, 1.0)
        self.biomass      = np.clip(self.biomass, 0.0, 5.0)
        self.local_toxin  = np.clip(self.local_toxin, 0.0, 1.0)

        # 5. Compute reward
        new_total_biomass  = float(self.biomass.sum())
        biomass_delta      = new_total_biomass - prev_total_biomass
        colony_coverage    = float(np.mean(self.biomass > 0.05))
        toxin_level        = float(self.global_state[5])
        ph_penalty         = abs(float(self.global_state[4]) - PH_OPTIMAL) * W_PH
        action_cost        = ACTION_COSTS[action]
        self.total_action_cost += action_cost

        reward = self._compute_reward(
            biomass_delta, colony_coverage, toxin_level, ph_penalty, action_cost
        )
        self.prev_coverage = colony_coverage

        self.step_count     += 1
        self.episode_reward += reward
        self.history.append(self._state_dict(action=action, reward=reward))

        # 6. Termination
        terminated = bool(colony_coverage < EXTINCTION_COVERAGE)
        truncated  = bool(self.step_count >= self.max_steps)

        info = self._info()
        if terminated or truncated:
            info["episode"] = {
                "r":              round(self.episode_reward, 3),
                "l":              self.step_count,
                "final_biomass":  round(new_total_biomass, 4),
                "final_coverage": round(colony_coverage, 4),
                "type_b_cells":   int(np.sum(
                    (self.cell_type == TYPE_B) & (self.biomass > 0.05)
                )),
                "reward_mode":    self.reward_mode,
            }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        print(self._render_ansi())

    def close(self):
        pass

    # -------------------------------------------------------------------------
    # Reward computation
    # -------------------------------------------------------------------------

    def _compute_reward(self, biomass_delta, colony_coverage,
                        toxin_level, ph_penalty, action_cost):
        """
        Compute reward based on selected reward_mode.

        multi_objective : balances biomass, coverage, toxin, cost, pH.
        biomass_only    : rewards biomass growth minus cost only.
        turbidity       : rewards colony coverage as indirect yield proxy,
                          mirroring real bioreactor optical density measurement.
        turbidity_delta : rewards change in colony coverage since the previous
                          step, minus cost.
        efficiency      : rewards biomass gained per unit of total cost spent.
        sustainability  : like multi_objective but double toxin penalty.
        """
        if self.reward_mode == "multi_objective":
            return (
                W_BIOMASS  * biomass_delta
              + W_COVERAGE * colony_coverage
              - W_COST     * action_cost
              - W_TOXIN    * toxin_level
              - ph_penalty
            )
        elif self.reward_mode == "biomass_only":
            return biomass_delta * 10.0 - action_cost
        elif self.reward_mode == "turbidity":
            return colony_coverage * 10.0 - action_cost * 0.5
        elif self.reward_mode == "turbidity_delta":
            # NOTE (intentional, kept for reproducibility): prev_coverage is
            # reset to 0.0 in reset(), so on step 1 of every episode this
            # reduces to (colony_coverage - 0) * 10 - cost*0.5, i.e. exactly
            # the plain "turbidity" reward for that one step only. From step
            # 2 onward it is a genuine coverage delta. Do not "fix" this
            # first-step artefact without a new env version.
            return (colony_coverage - self.prev_coverage) * 10.0 - action_cost * 0.5
        elif self.reward_mode == "efficiency":
            if self.total_action_cost > 0:
                return (self.biomass.sum() / self.total_action_cost) * 0.1
            else:
                return biomass_delta * 10.0
        elif self.reward_mode == "sustainability":
            return (
                W_BIOMASS  * biomass_delta
              + W_COVERAGE * colony_coverage
              - W_COST     * action_cost
              - W_TOXIN    * toxin_level * 2.0
              - ph_penalty
              - (1.0 - colony_coverage) * 1.0
            )
        else:
            raise ValueError(
                f"Unknown reward_mode: '{self.reward_mode}'. "
                f"Choose from: multi_objective, biomass_only, "
                f"turbidity, turbidity_delta, efficiency, sustainability"
            )

    # -------------------------------------------------------------------------
    # Action application
    # -------------------------------------------------------------------------

    def _apply_action(self, action: int):
        """Apply the RL agent's chosen action to global conditions."""
        g = self.global_state
        d = ACTION_DELTA

        if   action == 0:  pass
        elif action == 1:  g[0] = min(1.0, g[0] + d)
        elif action == 2:  g[0] = max(0.0, g[0] - d)
        elif action == 3:  g[2] = min(1.0, g[2] + d)
        elif action == 4:  g[2] = max(0.0, g[2] - d)
        elif action == 5:  g[3] = min(1.0, g[3] + 0.15)
        elif action == 6:  g[4] = min(1.0, g[4] + d)
        elif action == 7:
            # Filter must reduce local_toxin directly since _global_dynamics
            # recalculates g[5] from mean_local_toxin every step
            self.local_toxin = np.maximum(0.0, self.local_toxin - TOXIN_FILTER_BOOST)
        elif action == 8:  g[6] = max(0.0, g[6] - RADIATION_SHIELD)
        elif action == 9:  g[7] = min(1.0, g[7] + d)
        elif action == 10: self.pulse_steps_remaining = PULSE_DURATION

    # -------------------------------------------------------------------------
    # CA rules (vectorised)
    # -------------------------------------------------------------------------

    def _moore_sum(self, grid: np.ndarray) -> np.ndarray:
        """
        Fast Moore neighbourhood sum using NumPy roll operations.
        Returns the sum of the 8 neighbours for every cell simultaneously.
        Replaces the inner loop that called _count_living_neighbours per cell.
        """
        s = np.zeros_like(grid)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                s += np.roll(np.roll(grid, di, axis=0), dj, axis=1)
        return s

    def _ca_step(self):
        """
        Apply all active CA local rules to every cell simultaneously.

        Fully vectorised using NumPy array operations -- no Python loops
        over individual cells. Rules are applied to copies of the current
        arrays so every cell sees the same state before any updates.
        """
        g = self.global_state
        # temp and the scalar global toxin value are unpacked but unused here:
        # growth multipliers never depend on temperature (see TEMP_DRIFT_RATE
        # note above), and toxin logic in this function reads self.local_toxin
        # / self.world_config["toxin"] directly, not this scalar.
        light, _temp, co2, nutrients, ph, _toxin, radiation, humidity = g

        # -- Precompute global growth multipliers ----------------------------
        light_mult    = self._bell_curve_arr(light, LIGHT_SWEET_LOW, LIGHT_SWEET_HIGH, 1.4)
        co2_mult      = self._bell_curve_arr(co2,   CO2_SWEET_LOW,   CO2_SWEET_HIGH,   1.3)
        ph_mult       = self._bell_curve_arr(ph,     PH_OPTIMAL-0.15, PH_OPTIMAL+0.15, 1.2)
        hum_mult      = self._bell_curve_arr(humidity, HUMIDITY_OPTIMAL-0.2,
                                             HUMIDITY_OPTIMAL+0.2, 1.2)
        nutrient_mult = float(nutrients) ** 0.5
        rad_penalty_A = 1.0 - float(radiation) * 0.4   # Type A only

        pulse_mult = PULSE_SPREAD_BOOST if self.pulse_steps_remaining > 0 else 1.0

        # Working copies -- all updates written here, not to self arrays
        bio_new  = self.biomass.copy()
        tox_new  = self.local_toxin.copy()
        age_new  = self.cell_age.copy()
        type_new = self.cell_type.copy()

        # Boolean masks
        alive    = self.biomass > 0.05
        is_B     = (self.cell_type == TYPE_B)
        is_A     = alive & ~is_B

        # -- Rule 4: Toxin kill check ----------------------------------------
        if self.world_config["toxin"]:
            toxic_kill = alive & (self.local_toxin > TOXIN_KILL_THRESH)
            bio_new[toxic_kill] = np.maximum(0.0, self.biomass[toxic_kill] - 0.03)
            # Cells killed by toxin skip all other rules this step
            skip = toxic_kill
        else:
            skip = np.zeros((GRID_H, GRID_W), dtype=bool)

        # -- Rule 1: Photoinhibition (Type A only) ---------------------------
        if self.world_config["photoinhibition"]:
            photo = (is_A & (light > PHOTOINHIBITION_LIGHT)
                     & (co2 > PHOTOINHIBITION_CO2) & ~skip)
            bio_new[photo] = np.maximum(
                0.0, self.biomass[photo] + PHOTOINHIBITION_DAMAGE
            )
            if self.world_config["toxin"]:
                tox_new[photo] = np.minimum(1.0, self.local_toxin[photo] + 0.005)
            age_new[photo] += 1
            skip = skip | photo

        active = alive & ~skip   # cells that will grow normally this step

        # -- Rule 5: Colony support ------------------------------------------
        if self.world_config["colony_support"] and active.any():
            n_alive_neighbours = self._moore_sum(alive.astype(np.float32))
            colony_mod = np.ones((GRID_H, GRID_W), dtype=np.float32)
            colony_mod[active & (n_alive_neighbours >= COLONY_SUPPORT_THRESH)] = COLONY_SUPPORT_BONUS
            colony_mod[active & (n_alive_neighbours == ISOLATION_THRESH)]       = ISOLATION_PENALTY
        else:
            colony_mod = np.ones((GRID_H, GRID_W), dtype=np.float32)

        # -- Rule 1 (continued): Base growth ---------------------------------
        # Type A cells: full growth rate, penalised by radiation
        # Type B cells: 70% growth rate, immune to radiation and photoinhibition
        if active.any():
            tox_slow = (self.local_toxin > TOXIN_SLOW_THRESH) if self.world_config["toxin"] \
                       else np.zeros((GRID_H, GRID_W), dtype=bool)
            tox_mod = np.where(tox_slow, 0.5, 1.0)

            # Growth for active Type A cells
            active_A = active & is_A
            if active_A.any():
                growth_A = (BASE_GROWTH_RATE * light_mult * co2_mult * ph_mult
                            * nutrient_mult * hum_mult * rad_penalty_A
                            * tox_mod * colony_mod)
                bio_new[active_A]  = np.minimum(5.0, self.biomass[active_A] + growth_A[active_A])
                age_new[active_A] += 1
                if self.world_config["toxin"]:
                    tox_new[active_A] = np.minimum(
                        1.0,
                        self.local_toxin[active_A]
                        + growth_A[active_A] * TOXIN_PRODUCTION_A / BASE_GROWTH_RATE
                    )

            # Growth for active Type B cells (immune to radiation, 70% rate)
            active_B = active & is_B
            if active_B.any():
                growth_B = (BASE_GROWTH_RATE * 0.7 * light_mult * co2_mult * ph_mult
                            * nutrient_mult * hum_mult * tox_mod * colony_mod)
                bio_new[active_B]  = np.minimum(5.0, self.biomass[active_B] + growth_B[active_B])
                age_new[active_B] += 1
                if self.world_config["toxin"]:
                    tox_new[active_B] = np.minimum(
                        1.0,
                        self.local_toxin[active_B]
                        + growth_B[active_B] * TOXIN_PRODUCTION_B / BASE_GROWTH_RATE
                    )

        # -- Rule 3: Mutation ------------------------------------------------
        if self.world_config["mutation"]:
            can_mutate = (
                active & is_A
                & (self.cell_age >= MUTATION_AGE_THRESH)
                & (self.biomass  >= MUTATION_BIO_THRESH)
            )
            if can_mutate.any():
                rng_vals = self.np_random.random((GRID_H, GRID_W))
                type_new[can_mutate & (rng_vals < MUTATION_PROB)] = TYPE_B

        # -- Rule 2: Neighbour spreading -------------------------------------
        # Only cells with biomass > SPREAD_MIN_BIOMASS (0.10) spread.
        # This threshold is deliberately above the alive threshold (0.05)
        # so spreading cells cannot drop below alive threshold and kill themselves.
        if self.world_config["spreading"]:
            effective_spread = (SPREAD_RATE
                                * (1.0 + float(humidity) * HUMIDITY_SPREAD_MOD)
                                * pulse_mult)

            # Cells eligible to spread: alive and above spread minimum
            can_spread = bio_new > SPREAD_MIN_BIOMASS

            # For each direction, spread from donor to empty receiver
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1),
                           (-1,-1),(-1,1),(1,-1),(1,1)]:
                # Receiver position
                receiver_bio = np.roll(np.roll(bio_new, -di, axis=0), -dj, axis=1)
                empty        = receiver_bio <= 0.05

                # Donors: can spread and receiver is empty
                donors = can_spread & empty

                if not donors.any():
                    continue

                # Amount each donor spreads to this one neighbour
                # Divide by 8 so total spread across all directions = effective_spread
                amount = bio_new * (effective_spread / 8.0) * donors

                # Donor loses biomass
                bio_new -= amount

                # Receiver gains it (shift amount back to receiver positions)
                bio_new += np.roll(np.roll(amount, di, axis=0), dj, axis=1)

                # Toxin transfer
                if self.world_config["toxin"]:
                    tox_transfer = tox_new * 0.1 * donors
                    tox_new += np.roll(np.roll(tox_transfer, di, axis=0), dj, axis=1)

        # Write updated arrays back
        self.biomass     = bio_new
        self.local_toxin = tox_new
        self.cell_age    = age_new
        self.cell_type   = type_new

        if self.pulse_steps_remaining > 0:
            self.pulse_steps_remaining -= 1

    # -------------------------------------------------------------------------
    # Passive global dynamics
    # -------------------------------------------------------------------------

    def _global_dynamics(self):
        """
        Update global conditions passively each step.

        When no_drifts=True, all passive drifts are skipped except:
          - Radiation noise (always active -- models genuine measurement noise)
          - Colony-driven CO2 drawdown, nutrient consumption, and pH acidification
            (these depend on what the colony does, not external environment)
          - Toxin recalculation (depends on local_toxin which CA step just updated)
        """
        g             = self.global_state
        mean_biomass  = float(self.biomass.sum()) / N_CELLS

        if not self.no_drifts:
            # Temperature: passive drift toward cold snap
            g[1] += (TEMP_DRIFT_TARGET - g[1]) * TEMP_DRIFT_RATE

            # CO2: natural outgassing + colony drawdown
            g[2] = max(0.0, g[2] - CO2_NATURAL_DRIFT - mean_biomass * 0.003)

            # Nutrients: degradation rate increases over episode time
            degradation = (NUTRIENT_DEGRADATION_BASE
                           + NUTRIENT_DEGRADATION_RATE * self.step_count)
            g[3] = max(0.0, g[3] - degradation - mean_biomass * NUTRIENT_UPTAKE * 0.1)

            # pH: environmental acidification + colony acidification
            g[4] = max(0.0, g[4] - PH_ENV_DRIFT - mean_biomass * PH_ACIDIFICATION * 0.1)

            # Light: sinusoidal day/night cycle
            cycle_offset = LIGHT_CYCLE_AMPLITUDE * np.sin(
                2 * np.pi * self.step_count / LIGHT_CYCLE_PERIOD
            )
            g[0] = np.clip(g[0] + cycle_offset * 0.02, 0.0, 1.0)

            # Humidity: drifts toward moderate baseline
            g[7] += (0.4 - g[7]) * 0.01

        else:
            # no_drifts mode: only colony-driven changes remain
            g[2] = max(0.0, g[2] - mean_biomass * 0.003)
            g[3] = max(0.0, g[3] - mean_biomass * NUTRIENT_UPTAKE * 0.1)
            g[4] = max(0.0, g[4] - mean_biomass * PH_ACIDIFICATION * 0.1)

        # Radiation: always active regardless of no_drifts
        g[6] = np.clip(
            g[6] + self.np_random.uniform(-RADIATION_NOISE, RADIATION_NOISE),
            0.0, 1.0
        )

        # Toxin: recalculate from mean local toxin with natural decay
        g[5] = np.clip(float(self.local_toxin.mean()) - TOXIN_DECAY, 0.0, 1.0)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the 13-dimensional observation vector for the RL agent."""
        living_mask     = self.biomass > 0.05
        n_living        = max(1, int(living_mask.sum()))

        mean_biomass    = float(self.biomass.mean())
        colony_coverage = float(living_mask.mean())

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_toxin  = float(self.local_toxin[living_mask].mean()) \
                          if n_living > 1 else 0.0
            type_b_frac = float(
                np.sum((self.cell_type == TYPE_B) & living_mask) / n_living
            )
            mean_age    = float(self.cell_age[living_mask].mean() / self.max_steps) \
                          if n_living > 1 else 0.0

        return np.concatenate([
            self.global_state,
            np.array([mean_biomass, colony_coverage, mean_toxin,
                      type_b_frac, mean_age], dtype=np.float32)
        ]).astype(np.float32)

    def _info(self) -> dict:
        """Diagnostic info dict for debugging and SB3 logging."""
        living_mask = self.biomass > 0.05
        g = self.global_state
        return {
            "step":             self.step_count,
            "colony_coverage":  round(float(living_mask.mean()), 4),
            "total_biomass":    round(float(self.biomass.sum()), 4),
            "type_b_cells":     int(np.sum((self.cell_type == TYPE_B) & living_mask)),
            "pulse_active":     bool(self.pulse_steps_remaining > 0),
            "photoinhibition":  bool(g[0] > PHOTOINHIBITION_LIGHT
                                     and g[2] > PHOTOINHIBITION_CO2),
            "toxin_warning":    bool(g[5] > TOXIN_SLOW_THRESH),
            "nutrient_warning": bool(g[3] < 0.15),
            "reward_mode":      self.reward_mode,
            "no_drifts":        self.no_drifts,
        }

    def _state_dict(self, action=None, reward=None) -> dict:
        """Snapshot current state for history logging."""
        g      = self.global_state
        living = self.biomass > 0.05
        d = {
            "step":          self.step_count,
            "light":         round(float(g[0]), 4),
            "temp":          round(float(g[1]), 4),
            "co2":           round(float(g[2]), 4),
            "nutrients":     round(float(g[3]), 4),
            "ph":            round(float(g[4]), 4),
            "toxin":         round(float(g[5]), 4),
            "radiation":     round(float(g[6]), 4),
            "humidity":      round(float(g[7]), 4),
            "total_biomass": round(float(self.biomass.sum()), 4),
            "coverage":      round(float(living.mean()), 4),
            "type_b_count":  int(np.sum((self.cell_type == TYPE_B) & living)),
        }
        if action is not None: d["action"] = action
        if reward is not None: d["reward"] = round(float(reward), 4)
        return d

    def _render_ansi(self) -> str:
        """ASCII render of the grid and global conditions."""
        g = self.global_state
        lines = [
            f"\n-- AlphaeWorld v2  Step {self.step_count} "
            f"[{self.reward_mode}{'  no_drifts' if self.no_drifts else ''}] --"
        ]
        for i in range(GRID_H):
            row = ""
            for j in range(GRID_W):
                bio = self.biomass[i, j]
                if bio <= 0.05:
                    row += ". "
                elif self.cell_type[i, j] == TYPE_B:
                    row += "B "
                elif bio > 0.6:
                    row += "# "
                elif bio > 0.3:
                    row += "* "
                else:
                    row += "o "
            lines.append("  " + row)

        names = ["Light","Temp ","CO2  ","Nutr ","pH   ","Toxin","Rad  ","Humid"]
        lines.append("")
        for name, val in zip(names, g):
            filled = int(val * 15)
            bar    = "#" * filled + "." * (15 - filled)
            lines.append(f"  {name}: [{bar}] {val:.3f}")

        living = self.biomass > 0.05
        lines.append(
            f"\n  Coverage: {living.mean()*100:.1f}%  "
            f"Total bio: {self.biomass.sum():.2f}  "
            f"Type B: {np.sum((self.cell_type==TYPE_B)&living)}  "
            f"Reward: {self.episode_reward:.2f}"
        )
        if self.pulse_steps_remaining > 0:
            lines.append(f"  PULSE ACTIVE ({self.pulse_steps_remaining} steps)")

        return "\n".join(lines)

    @staticmethod
    def _bell_curve_arr(x: float, lo: float, hi: float,
                        peak: float = 1.0) -> float:
        """
        Scalar bell curve multiplier centred on sweet spot [lo, hi].
        Returns peak at midpoint, decays to 0.05 beyond 2x the half-width.
        Used for precomputing global multipliers that apply to all cells equally.
        """
        mid   = (lo + hi) / 2.0
        width = (hi - lo) / 2.0
        dist  = abs(float(x) - mid)
        if dist >= width * 2:
            return 0.05
        t = dist / (width * 2)
        return peak * (0.5 + 0.5 * np.cos(np.pi * t))