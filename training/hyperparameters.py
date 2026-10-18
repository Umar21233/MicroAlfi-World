"""
hyperparameters.py -- The tuned hyperparameters actually used to train the
models under "Colab training results/" (and therefore results_verified/).

Confirmed by reading the `data` JSON stored inside one model.zip per
algorithm (see docs/VERIFICATION.md, Step 4): learning_rate=0.001 for all
three algorithms, DQN batch_size=64, PPO n_steps=2048, A2C n_steps=64,
gamma=0.99 for all three. Everything else was left at the Stable-Baselines3
default for that algorithm.

train_dqn.py / train_ppo.py / train_a2c.py default their CLI flags to
different values (learning_rate=0.0003, A2C n_steps=2048) -- their own
previous, untuned defaults -- so existing behaviour for any run already
done is unaffected. Pass --tuned to any of the three training scripts to
use the values below instead, for NEW runs only.
"""

TUNED = {
    "dqn": {"learning_rate": 0.001, "batch_size": 64, "gamma": 0.99},
    "ppo": {"learning_rate": 0.001, "n_steps": 2048, "gamma": 0.99},
    "a2c": {"learning_rate": 0.001, "n_steps": 64, "gamma": 0.99},
}
