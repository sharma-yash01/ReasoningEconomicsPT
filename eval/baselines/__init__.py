"""Baselines: dummy deterministic and LLM-backed strategies."""

from eval.baselines.dummy import (
    UniformBaseline,
    GreedyMaxBaseline,
    DifficultyOracleBaseline,
)
from eval.baselines.llm import (
    APIChatBaseline,
    LocalVLLMBaseline,
)

__all__ = [
    "UniformBaseline",
    "GreedyMaxBaseline",
    "DifficultyOracleBaseline",
    "APIChatBaseline",
    "LocalVLLMBaseline",
]
