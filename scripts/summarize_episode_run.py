#!/usr/bin/env python3
"""Summarize an episode-mode reward log into JSON + Markdown reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.episode_summary import write_episode_summary_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an episode-mode reward_log.jsonl.")
    parser.add_argument("log_path", type=Path, help="Path to reward_log.jsonl")
    args = parser.parse_args()

    json_path, md_path = write_episode_summary_outputs(args.log_path)
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    main()
