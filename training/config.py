"""Configuration objects for GRPO + OpenEnv training runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingRuntimeConfig:
    """Runtime controls used by training/grpo_train.py."""

    # Reward weighting (total signal path for now).
    alpha: float = 1.0
    beta: float = 0.0

    # Token generation controls.
    max_tokens_per_step: int = 2048
    min_tokens_per_step: int = 10

    # Mode detection controls.
    default_budget_mode: str = "hard"
    strict_budget_mode_metadata: bool = False

    # Reward logging controls.
    log_rewards: bool = True
    log_every_n_steps: int = 1
    reward_log_path: str = ""

    def normalized_default_mode(self) -> str:
        mode = str(self.default_budget_mode or "hard").strip().lower()
        if mode in {"hard", "soft"}:
            return mode
        return "hard"

    def resolved_reward_log_path(self, output_dir: str) -> str:
        if self.reward_log_path:
            return self.reward_log_path
        return str(Path(output_dir) / "reward_logs.jsonl")
