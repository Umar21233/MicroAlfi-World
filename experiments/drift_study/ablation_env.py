"""
ablation_env.py -- Drift study Part B.1.

AblationDriftEnv subclasses AlphaeWorldEnvV2 and overrides ONLY
_global_dynamics, so each of the six passive drifts (temperature, co2,
nutrients, ph, light, humidity) can be switched on or off independently.
Nothing else about the environment changes: reset(), step(), _apply_action,
_ca_step, reward computation, observation building, and RNG usage are all
inherited unchanged from AlphaeWorldEnvV2.

Design, read directly off AlphaeWorldEnvV2._global_dynamics:
    - Radiation noise is applied unconditionally in the original (both
      no_drifts branches), via exactly one self.np_random.uniform() call
      per step. It stays unconditional here too, called exactly once per
      step regardless of any flag, so the RNG stream a model/action
      sequence sees is identical whenever all six flags match one of the
      two original no_drifts settings -- this is what Part B.2's
      bit-exact equivalence test checks.
    - co2 / nutrients / ph each have a "colony-driven" component (driven by
      mean_biomass, present in BOTH original no_drifts branches) and a
      "passive drift" component (present only when no_drifts=False). Each
      flag here gates only that variable's passive component; the
      colony-driven component is always applied, exactly as the original.
    - temperature, light, and humidity have no colony-driven component in
      the original (they are untouched in the no_drifts=True branch), so
      each flag here simply gates the entire update for that variable.
    - Toxin recalculation (from local_toxin, not a "drift") is always
      applied, unconditionally, exactly as the original.

drift_flags with all six True reproduces no_drifts=False exactly.
drift_flags with all six False reproduces no_drifts=True exactly.
Any other combination is a genuinely new ablation condition this
environment did not exist to express before.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.alphae_world_env_v2 import (  # noqa: E402
    AlphaeWorldEnvV2,
    N_CELLS,
    TEMP_DRIFT_TARGET, TEMP_DRIFT_RATE,
    CO2_NATURAL_DRIFT,
    NUTRIENT_DEGRADATION_BASE, NUTRIENT_DEGRADATION_RATE, NUTRIENT_UPTAKE,
    PH_ENV_DRIFT, PH_ACIDIFICATION,
    LIGHT_CYCLE_AMPLITUDE, LIGHT_CYCLE_PERIOD,
    RADIATION_NOISE,
    TOXIN_DECAY,
)

DRIFT_NAMES = ("temperature", "co2", "nutrients", "ph", "light", "humidity")

ALL_ON  = {name: True for name in DRIFT_NAMES}
ALL_OFF = {name: False for name in DRIFT_NAMES}


class AblationDriftEnv(AlphaeWorldEnvV2):
    """
    AlphaeWorldEnvV2 with each passive drift individually toggleable via
    drift_flags. The no_drifts constructor argument is accepted but unused
    by this subclass's own logic (kept only so the base class's __init__
    and _info()/_render_ansi() labels still work); drift_flags is what
    actually controls _global_dynamics here.
    """

    def __init__(self, drift_flags: dict, **kwargs):
        missing = set(DRIFT_NAMES) - set(drift_flags)
        if missing:
            raise ValueError(f"drift_flags missing keys: {missing}")
        super().__init__(**kwargs)
        self.drift_flags = dict(drift_flags)

    def _global_dynamics(self):
        g = self.global_state
        f = self.drift_flags
        mean_biomass = float(self.biomass.sum()) / N_CELLS

        # Temperature: passive drift only, no colony-driven component.
        if f["temperature"]:
            g[1] += (TEMP_DRIFT_TARGET - g[1]) * TEMP_DRIFT_RATE

        # CO2: colony-driven drawdown always applies; natural outgassing
        # only when the co2 drift flag is on.
        g[2] = max(0.0, g[2]
                   - (CO2_NATURAL_DRIFT if f["co2"] else 0.0)
                   - mean_biomass * 0.003)

        # Nutrients: colony-driven uptake always applies; time-increasing
        # degradation only when the nutrients drift flag is on.
        degradation = (NUTRIENT_DEGRADATION_BASE
                        + NUTRIENT_DEGRADATION_RATE * self.step_count) if f["nutrients"] else 0.0
        g[3] = max(0.0, g[3] - degradation - mean_biomass * NUTRIENT_UPTAKE * 0.1)

        # pH: colony-driven acidification always applies; environmental
        # drift only when the ph drift flag is on.
        g[4] = max(0.0, g[4]
                   - (PH_ENV_DRIFT if f["ph"] else 0.0)
                   - mean_biomass * PH_ACIDIFICATION * 0.1)

        # Light: sinusoidal day/night cycle, no colony-driven component.
        if f["light"]:
            cycle_offset = LIGHT_CYCLE_AMPLITUDE * np.sin(
                2 * np.pi * self.step_count / LIGHT_CYCLE_PERIOD
            )
            g[0] = np.clip(g[0] + cycle_offset * 0.02, 0.0, 1.0)

        # Humidity: drift toward baseline, no colony-driven component.
        if f["humidity"]:
            g[7] += (0.4 - g[7]) * 0.01

        # Radiation: always active regardless of any flag -- exactly one
        # RNG draw per step, same as the original, in both branches.
        g[6] = np.clip(
            g[6] + self.np_random.uniform(-RADIATION_NOISE, RADIATION_NOISE),
            0.0, 1.0
        )

        # Toxin: recalculated from local_toxin, not a "drift" -- always
        # applied, exactly as the original.
        g[5] = np.clip(float(self.local_toxin.mean()) - TOXIN_DECAY, 0.0, 1.0)
