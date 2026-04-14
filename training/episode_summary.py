"""Helpers for summarizing episode-mode reward logs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def safe_mean(vals: list[float]) -> float | None:
    return mean(vals) if vals else None


def safe_std(vals: list[float]) -> float:
    return stdev(vals) if len(vals) >= 2 else 0.0


def load_reward_log(log_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _is_episode_record(record: dict[str, Any]) -> bool:
    return record.get("event") == "episode_end" or (
        "episode_reward" in record and "steps" in record
    )


def _normalize_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    final_observation = record.get("final_observation") or {}
    final_questions_remaining = int(
        record.get(
            "final_questions_remaining",
            final_observation.get("questions_remaining", 0),
        )
    )
    questions_completed = int(
        record.get("questions_completed", record.get("num_steps", len(record.get("steps", []))))
    )
    termination_reason = record.get("termination_reason")
    if not termination_reason:
        termination_reason = "env_done" if final_questions_remaining == 0 else "unknown"

    normalized = dict(record)
    normalized["episode_weighted_reward"] = float(
        record.get("episode_weighted_reward", record.get("episode_reward", 0.0))
    )
    normalized["questions_completed"] = questions_completed
    normalized["final_questions_remaining"] = final_questions_remaining
    normalized["total_completion_tokens"] = record.get("total_completion_tokens")
    normalized["total_tokens_serialized"] = record.get("total_tokens_serialized")
    normalized["episode_clipped"] = bool(record.get("episode_clipped", False))
    normalized["termination_reason"] = termination_reason
    return normalized


def summarize_episode_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [_normalize_episode_record(record) for record in records if _is_episode_record(record)]
    episodes.sort(key=lambda e: (e.get("seed", -1), e.get("episode_idx", -1), e.get("episode_id", "")))
    if not episodes:
        raise ValueError("No episode-end records found in reward log.")

    rewards = [float(e["episode_weighted_reward"]) for e in episodes]
    completed = [int(e.get("questions_completed", 0)) for e in episodes]
    completion_tokens = [
        float(e["total_completion_tokens"])
        for e in episodes
        if e.get("total_completion_tokens") is not None
    ]
    completion_rate = (
        sum(1 for e in episodes if int(e.get("final_questions_remaining", 1)) == 0) / len(episodes)
    )
    clipped_rate = sum(1 for e in episodes if e.get("episode_clipped", False)) / len(episodes)
    termination_reasons = Counter(str(e.get("termination_reason", "unknown")) for e in episodes)

    return {
        "num_episodes": len(episodes),
        "mean_reward": safe_mean(rewards),
        "std_reward": safe_std(rewards),
        "mean_questions_completed": safe_mean([float(v) for v in completed]),
        "completion_rate": completion_rate,
        "mean_completion_tokens": safe_mean(completion_tokens),
        "clipped_rate": clipped_rate,
        "termination_reasons": dict(termination_reasons),
        "episodes": episodes,
    }


def render_episode_summary_markdown(summary: dict[str, Any]) -> str:
    episodes = summary["episodes"]
    rows = "\n".join(
        f"| {e.get('seed', '?')} | {e.get('episode_weighted_reward', 0):.4f} | "
        f"{e.get('questions_completed', '?')} | {e.get('final_questions_remaining', '?')} | "
        f"{e.get('total_completion_tokens', '?')} | {e.get('total_tokens_serialized', '?')} | "
        f"{e.get('episode_clipped', '?')} | {e.get('termination_reason', '?')} |"
        for e in episodes
    )

    mean_reward = summary.get("mean_reward")
    mean_reward_str = f"{mean_reward:.4f}" if mean_reward is not None else "n/a"
    mean_tokens = summary.get("mean_completion_tokens")
    mean_tokens_str = f"{mean_tokens:.1f}" if mean_tokens is not None else "n/a"
    mean_completed = summary.get("mean_questions_completed")
    mean_completed_str = f"{mean_completed:.2f}" if mean_completed is not None else "n/a"

    return f"""# Episode Run Summary

## Aggregates
- Episodes: {summary['num_episodes']}
- Completion rate: {summary['completion_rate']:.0%}
- Mean reward: {mean_reward_str} ± {summary['std_reward']:.4f}
- Mean questions completed: {mean_completed_str}
- Mean completion tokens (model-only): {mean_tokens_str}
- Clipped rate: {summary['clipped_rate']:.0%}
- Termination reasons: {summary['termination_reasons']}

## Per-Episode
| seed | total_reward | questions_completed | final_questions_remaining | completion_tokens | serialized_tokens | clipped | termination_reason |
|------|-------------:|--------------------:|--------------------------:|------------------:|------------------:|:-------:|-------------------|
{rows}
"""


def write_episode_summary_outputs(log_path: Path) -> tuple[Path, Path]:
    summary = summarize_episode_records(load_reward_log(log_path))
    out_dir = log_path.parent
    json_path = out_dir / "episode_summary.json"
    md_path = out_dir / "episode_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(render_episode_summary_markdown(summary), encoding="utf-8")
    return json_path, md_path
