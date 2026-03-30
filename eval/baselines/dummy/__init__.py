"""Deterministic baselines for smoke testing env/reward behavior."""

from eval.baselines.dummy.uniform import UniformBaseline
from eval.baselines.dummy.greedy_max import GreedyMaxBaseline
from eval.baselines.dummy.difficulty_oracle import DifficultyOracleBaseline

__all__ = [
    "UniformBaseline",
    "GreedyMaxBaseline",
    "DifficultyOracleBaseline",
]
