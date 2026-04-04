"""Configuration objects for GRPO + OpenEnv training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRuntimeConfig:
    """Runtime controls used by training/grpo_train.py.

    After migrating to environment_factory, only ``alpha`` and the reward-
    logging fields are actively used by grpo_train.  The remaining fields
    are kept for backward compatibility with other consumers (e.g.
    openenv_runtime.resolve_budget_mode_from_observation).
    """

    # Reward weighting (total signal path for now).
    alpha: float = 1.0
    beta: float = 0.0  # unused by environment_factory path

    # Token generation controls -- unused after environment_factory migration;
    # the trainer now controls generation length via max_completion_length.
    max_tokens_per_step: int = 2048
    min_tokens_per_step: int = 10

    # Mode detection controls -- unused after environment_factory migration;
    # the env server enforces hard-cap mode directly.
    default_budget_mode: str = "hard"
    strict_budget_mode_metadata: bool = False

    # Reward logging controls.
    log_rewards: bool = True
    log_every_n_steps: int = 1
    reward_log_path: str = ""

    def normalized_default_mode(self):
        mode = str(self.default_budget_mode or "hard").strip().lower()
        if mode in {"hard", "soft"}:
            return mode
        return "hard"

    def resolved_reward_log_path(self, output_dir: str):
        if self.reward_log_path:
            return self.reward_log_path
        return str(Path(output_dir) / "reward_logs.jsonl")
