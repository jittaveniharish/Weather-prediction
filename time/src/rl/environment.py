"""
Gymnasium environment for temperature correction via residual actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from gymnasium import Env, spaces

from .. import config


def compose_observation(context: np.ndarray, baseline_seq: np.ndarray) -> np.ndarray:
    ctx = context.astype(np.float32)
    baseline_seq = baseline_seq.astype(np.float32)
    ctx_mean = np.mean(ctx)
    ctx_std = np.std(ctx) + 1e-3
    norm_ctx = (ctx - ctx_mean) / ctx_std
    baseline_norm = (baseline_seq - ctx_mean) / ctx_std
    deltas = np.diff(np.concatenate([ctx[-1:], baseline_seq]))
    features = np.concatenate(
        [
            norm_ctx,
            baseline_norm,
            deltas,
            np.array([ctx_mean, ctx_std], dtype=np.float32),
        ]
    )
    return features.astype(np.float32)


@dataclass(frozen=True)
class TransitionDataset:
    contexts: np.ndarray  # shape: (N, window)
    baselines: np.ndarray  # shape: (N, horizon)
    targets: np.ndarray  # shape: (N, horizon)
    start_indices: np.ndarray  # shape: (N,)

    def __post_init__(self) -> None:
        assert self.contexts.ndim == 2, "contexts must be 2D"
        assert self.baselines.ndim == 2, "baselines must be 2D"
        assert self.targets.ndim == 2, "targets must be 2D"
        assert len(self.contexts) == len(self.baselines) == len(self.targets) == len(
            self.start_indices
        ), "Dataset components must share the same first dimension."

    @property
    def horizon(self) -> int:
        return self.baselines.shape[1]


class TemperatureCorrectionEnv(Env):
    """
    Single-step environment where the agent outputs a residual correction sequence.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset: TransitionDataset):
        super().__init__()
        self.dataset = dataset
        self.window = dataset.contexts.shape[1]
        self.horizon = dataset.horizon
        obs_dim = self.window + self.horizon * 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.horizon,), dtype=np.float32
        )
        self._index = 0

    def _make_observation(self, idx: int) -> np.ndarray:
        ctx = self.dataset.contexts[idx]
        baseline_seq = self.dataset.baselines[idx]
        return compose_observation(ctx, baseline_seq)

    def observation_for(self, idx: int) -> np.ndarray:
        return self._make_observation(idx)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._index = self.np_random.integers(0, len(self.dataset.contexts))
        obs = self._make_observation(self._index)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        baseline = self.dataset.baselines[self._index]
        target = self.dataset.targets[self._index]
        correction = np.asarray(action, dtype=np.float32) * config.RL_ACTION_SCALE
        prediction = baseline + correction
        reward = -float(np.mean(np.abs(target - prediction)))
        obs = self._make_observation(self._index)
        terminated = True
        truncated = False
        info = {
            "baseline": baseline,
            "corrected": prediction,
            "target": target,
            "correction": correction,
        }
        return obs, reward, terminated, truncated, info


