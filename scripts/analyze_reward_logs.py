#!/usr/bin/env python3
"""Analyze GRPO ``reward_logs.jsonl`` for reward distributions and env health signals.

Parses the format written by ``training/grpo_train.py`` (one JSON object per line):
episode-level ``episode_reward``, ``num_steps``, ``steps[*].raw_step_reward``, etc.

Usage:
  pip install -r requirements.analysis.txt
  python scripts/analyze_reward_logs.py path/to/reward_logs.jsonl --out-dir ./reward_analysis

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skip line {lineno}: {e}", file=sys.stderr)
    return rows


def episodes_to_dataframe(episodes: list[dict]) -> pd.DataFrame:
    records = []
    for i, ep in enumerate(episodes):
        steps = ep.get("steps") or []
        first = steps[0] if steps else {}
        last_step = steps[-1] if steps else {}
        fo = ep.get("final_observation") or {}
        records.append({
            "episode_idx": i,
            "episode_id": ep.get("episode_id"),
            "timestamp_utc": ep.get("timestamp_utc"),
            "episode_reward": float(ep.get("episode_reward", 0.0)),
            "num_steps": int(ep.get("num_steps", 0)),
            "first_questions_remaining": first.get("questions_remaining_before"),
            "first_remaining_budget": first.get("remaining_budget_before"),
            "first_raw_step_reward": float(first.get("raw_step_reward", 0.0) or 0.0),
            "last_done": bool(last_step.get("done_after_step", False)),
            "final_obs_step_idx": fo.get("step_idx"),
            "final_obs_q_rem": fo.get("questions_remaining"),
            "final_obs_remaining_budget": fo.get("remaining_budget"),
            "final_history_len": len(fo.get("history") or []),
        })
    return pd.DataFrame(records)


def steps_to_dataframe(episodes: list[dict]) -> pd.DataFrame:
    rows = []
    for i, ep in enumerate(episodes):
        eid = ep.get("episode_id")
        for s in ep.get("steps") or []:
            rows.append({
                "episode_idx": i,
                "episode_id": eid,
                "step_index": s.get("step_index"),
                "raw_step_reward": float(s.get("raw_step_reward", 0.0) or 0.0),
                "scaled_step_reward": float(s.get("scaled_step_reward", 0.0) or 0.0),
                "questions_remaining_before": s.get("questions_remaining_before"),
                "remaining_budget_before": s.get("remaining_budget_before"),
                "done_after_step": bool(s.get("done_after_step", False)),
            })
    return pd.DataFrame(rows)


def print_summary(ep_df: pd.DataFrame, st_df: pd.DataFrame) -> None:
    n_ep = len(ep_df)
    n_st = len(st_df)
    print("=== Reward log summary ===")
    print(f"Episodes: {n_ep}  |  Step rows: {n_st}")
    if n_ep == 0:
        return

    er = ep_df["episode_reward"]
    print("\n--- Episode reward ---")
    print(er.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
    print(f"Fraction episode_reward == 0: {(er == 0).mean():.3f}")
    print(f"Fraction episode_reward < 0:  {(er < 0).mean():.3f}")

    print("\n--- Episode length (num_steps) ---")
    print(ep_df["num_steps"].value_counts().sort_index().head(20).to_string())
    single = (ep_df["num_steps"] == 1).sum()
    print(f"Fraction num_steps == 1: {single / n_ep:.3f}  ({single}/{n_ep})")

    # Heuristic: immediate abort (common env/protocol issue)
    mask_abort = (
        (ep_df["num_steps"] == 1)
        & (ep_df["first_raw_step_reward"] == 0)
        & (ep_df["first_questions_remaining"] == 10)
    )
    print(f"Fraction 'single-step zero @ Q=10': {mask_abort.mean():.3f}  (possible early env abort)")

    if len(st_df):
        print("\n--- Per-step raw_step_reward (all steps) ---")
        print(st_df["raw_step_reward"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())
        print(f"Fraction raw_step_reward == 0: {(st_df['raw_step_reward'] == 0).mean():.3f}")

    print("\n--- Final observation sanity (last row per episode) ---")
    empty_q = (ep_df["final_obs_q_rem"] == 0) & (ep_df["final_obs_step_idx"] == 0)
    print(f"Fraction final_obs empty-like (q_rem=0 & step_idx=0): {empty_q.mean():.3f}")


def plot_all(ep_df: pd.DataFrame, st_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    # Episode reward distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(ep_df["episode_reward"], kde=True, ax=ax, bins=min(40, max(10, len(ep_df) // 5)))
    ax.set_title("Episode reward distribution")
    ax.set_xlabel("episode_reward (sum of scaled step rewards)")
    fig.tight_layout()
    fig.savefig(out_dir / "01_episode_reward_hist.png", dpi=150)
    plt.close(fig)

    # num_steps distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    vc = ep_df["num_steps"].value_counts().sort_index()
    top = vc.head(25)
    sns.barplot(x=top.index.astype(int), y=top.values, ax=ax, color="steelblue")
    ax.set_title("Episodes by num_steps (top 25 lengths)")
    ax.set_xlabel("num_steps")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "02_num_steps_count.png", dpi=150)
    plt.close(fig)

    # Episode reward vs num_steps
    fig, ax = plt.subplots(figsize=(8, 4))
    cap = int(ep_df["num_steps"].quantile(0.99)) if len(ep_df) > 5 else int(ep_df["num_steps"].max())
    cap = max(cap, 1)
    sub = ep_df[ep_df["num_steps"] <= cap]
    sns.boxplot(data=sub, x="num_steps", y="episode_reward", ax=ax, color="lightcyan")
    ax.set_title(f"Episode reward vs num_steps (≤{cap} p99 cap)")
    fig.tight_layout()
    fig.savefig(out_dir / "03_episode_reward_vs_num_steps_box.png", dpi=150)
    plt.close(fig)

    # Time / order index
    fig, ax = plt.subplots(figsize=(9, 4))
    window = max(1, min(25, len(ep_df) // 20))
    rolling = ep_df["episode_reward"].rolling(window=window, min_periods=1).mean()
    ax.plot(ep_df["episode_idx"], ep_df["episode_reward"], alpha=0.25, label="episode_reward")
    ax.plot(ep_df["episode_idx"], rolling, color="darkred", label=f"rolling mean (w={window})")
    ax.set_xlabel("episode index (file order)")
    ax.set_ylabel("reward")
    ax.legend()
    ax.set_title("Episode reward over training order")
    fig.tight_layout()
    fig.savefig(out_dir / "04_episode_reward_trajectory.png", dpi=150)
    plt.close(fig)

    if len(st_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(st_df["raw_step_reward"], kde=True, ax=ax, bins=40)
        ax.set_title("Per-step raw_step_reward (from env, pre-alpha)")
        ax.set_xlabel("raw_step_reward")
        fig.tight_layout()
        fig.savefig(out_dir / "05_raw_step_reward_hist.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.scatterplot(
            data=st_df.sample(min(5000, len(st_df))) if len(st_df) > 5000 else st_df,
            x="remaining_budget_before",
            y="raw_step_reward",
            alpha=0.3,
            ax=ax,
        )
        ax.set_title("raw_step_reward vs remaining_budget_before (subsample if large)")
        fig.tight_layout()
        fig.savefig(out_dir / "06_step_reward_vs_budget.png", dpi=150)
        plt.close(fig)

    # Diagnostic flags
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = []
    vals = []
    labels.append("episode_reward == 0")
    vals.append((ep_df["episode_reward"] == 0).mean())
    labels.append("num_steps == 1")
    vals.append((ep_df["num_steps"] == 1).mean())
    labels.append("single-step + raw=0 + Q=10")
    m = (
        (ep_df["num_steps"] == 1)
        & (ep_df["first_raw_step_reward"] == 0)
        & (ep_df["first_questions_remaining"] == 10)
    )
    vals.append(m.mean())
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["coral", "steelblue", "darkred"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of episodes")
    ax.set_title("Health / abort heuristics (for reward redesign)")
    fig.tight_layout()
    fig.savefig(out_dir / "07_diagnostic_fractions.png", dpi=150)
    plt.close(fig)

    print(f"\nFigures written to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reward_logs.jsonl from GRPO training.")
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to reward_logs.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reward_log_analysis"),
        help="Directory for PNG figures (created if missing)",
    )
    args = parser.parse_args()

    if not args.jsonl_path.is_file():
        print(f"Error: file not found: {args.jsonl_path}", file=sys.stderr)
        sys.exit(1)

    episodes = load_jsonl(args.jsonl_path)
    if not episodes:
        print("No valid rows; exiting.", file=sys.stderr)
        sys.exit(2)

    ep_df = episodes_to_dataframe(episodes)
    st_df = steps_to_dataframe(episodes)
    print_summary(ep_df, st_df)
    plot_all(ep_df, st_df, args.out_dir)


if __name__ == "__main__":
    main()
