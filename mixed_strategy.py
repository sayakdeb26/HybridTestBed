#!/usr/bin/env python3
"""
Anti-Gravity Lab — Mixed Strategy

Combines EWC penalty + Experience Replay for continual learning.
Inherits EWC logic from EWCStrategy (DRY) and adds a replay buffer
on top.
"""
import time
from typing import Dict

import numpy as np

from .base_strategy import StrategyConfig
from .ewc_strategy import EWCStrategy
from .replay_strategy import ReplayBuffer


class MixedStrategy(EWCStrategy):
    """Combined EWC + Replay strategy.

    Inherits all EWC mechanics (Online EWC with running Fisher) from
    ``EWCStrategy``.  On top of that, maintains a class-balanced
    ``ReplayBuffer`` whose samples are concatenated to each training
    batch before calling the parent's ``train_on_batch``.

    The effective EWC lambda seen by the penalty term is
    ``ewc_lambda * ewc_weight`` — this keeps the weighting knob from
    the original design without duplicating code.
    """

    def __init__(self, config: StrategyConfig,
                 ewc_lambda: float = 0.3,
                 ewc_weight: float = 0.4,
                 replay_weight: float = 0.6,
                 buffer_size: int = 300,
                 fisher_samples: int = 150,
                 online_alpha: float = 0.5):
        # Pass effective lambda = ewc_lambda * ewc_weight to EWCStrategy
        super().__init__(
            config,
            ewc_lambda=ewc_lambda * ewc_weight,
            fisher_samples=fisher_samples,
            online_alpha=online_alpha,
        )
        self.name = "mixed"
        self.replay_weight = replay_weight

        # Replay components (class-balanced)
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_size, balanced=True,
            weak_class_ids=(2, 5),
        )

    def train_on_batch(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train with EWC penalty + augmented replay data.

        Concatenates replay samples onto the incoming batch, then
        delegates the entire training loop to ``EWCStrategy.train_on_batch``.
        """
        # Mix replay samples into the training batch
        replay_n = int(len(X) * self.replay_weight)
        if self.replay_buffer.size() > 0 and replay_n > 0:
            X_replay, y_replay = self.replay_buffer.sample(replay_n)
            if len(X_replay) > 0:
                # Augment replay data to prevent overfitting
                if getattr(self.config, 'augment_replay', True):
                    X_replay = self._augment_time_series(X_replay)
                X = np.concatenate([X, X_replay], axis=0)
                y = np.concatenate([y, y_replay], axis=0)

        # Delegate to parent (handles EWC penalty, scheduler, etc.)
        metrics = super().train_on_batch(X, y)
        metrics["buffer_size"] = self.replay_buffer.size()
        return metrics

    def on_task_end(self, X: np.ndarray, y: np.ndarray):
        """Compute Online EWC Fisher + add samples to replay buffer."""
        # EWC consolidation (parent handles running Fisher merge)
        super().on_task_end(X, y)
        # Replay buffer update
        self.replay_buffer.add_batch(X, y)

    def save_state(self, filepath: str):
        """Save EWC and Replay Buffer state to disk."""
        import torch
        state = {
            "fisher": {k: v.cpu() for k, v in self.fisher.items()},
            "optpar": {k: v.cpu() for k, v in self.optpar.items()},
            "replay_X": self.replay_buffer.X_buffer,
            "replay_y": self.replay_buffer.y_buffer
        }
        torch.save(state, filepath)

    def load_state(self, filepath: str):
        """Load EWC and Replay Buffer state from disk."""
        import torch
        import os
        if not os.path.exists(filepath):
            return False
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.fisher = {k: v.to(self.device) for k, v in state["fisher"].items()}
        self.optpar = {k: v.to(self.device) for k, v in state["optpar"].items()}
        self.replay_buffer.X_buffer = state["replay_X"]
        self.replay_buffer.y_buffer = state["replay_y"]
        return True

