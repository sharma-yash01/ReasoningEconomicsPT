"""Unit tests for episode summary helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.episode_summary import (
    render_episode_summary_markdown,
    summarize_episode_records,
    write_episode_summary_outputs,
)


class TestEpisodeSummary(unittest.TestCase):
    def test_summarize_new_schema(self):
        records = [
            {
                "event": "episode_end",
                "seed": 0,
                "episode_idx": 0,
                "episode_weighted_reward": 0.75,
                "questions_completed": 4,
                "final_questions_remaining": 0,
                "total_completion_tokens": 167,
                "total_tokens_serialized": 772,
                "episode_clipped": False,
                "termination_reason": "env_done",
            },
            {
                "event": "episode_end",
                "seed": 1,
                "episode_idx": 0,
                "episode_weighted_reward": -0.25,
                "questions_completed": 3,
                "final_questions_remaining": 1,
                "total_completion_tokens": 120,
                "total_tokens_serialized": 640,
                "episode_clipped": True,
                "termination_reason": "max_episode_turns",
            },
        ]

        summary = summarize_episode_records(records)
        self.assertEqual(summary["num_episodes"], 2)
        self.assertEqual(summary["termination_reasons"]["env_done"], 1)
        self.assertEqual(summary["termination_reasons"]["max_episode_turns"], 1)
        self.assertAlmostEqual(summary["completion_rate"], 0.5)
        self.assertAlmostEqual(summary["clipped_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_completion_tokens"], 143.5)

        markdown = render_episode_summary_markdown(summary)
        self.assertIn("Episode Run Summary", markdown)
        self.assertIn("env_done", markdown)
        self.assertIn("max_episode_turns", markdown)

    def test_summarize_legacy_schema(self):
        records = [
            {
                "episode_id": "abc",
                "episode_reward": 1.0,
                "num_steps": 4,
                "steps": [{"step_index": 1}, {"step_index": 2}, {"step_index": 3}, {"step_index": 4}],
                "final_observation": {"questions_remaining": 0},
            }
        ]

        summary = summarize_episode_records(records)
        self.assertEqual(summary["num_episodes"], 1)
        self.assertEqual(summary["episodes"][0]["questions_completed"], 4)
        self.assertEqual(summary["episodes"][0]["termination_reason"], "env_done")

    def test_write_outputs(self):
        records = [
            {
                "event": "episode_end",
                "seed": 0,
                "episode_idx": 0,
                "episode_weighted_reward": 0.5,
                "questions_completed": 4,
                "final_questions_remaining": 0,
                "total_completion_tokens": 100,
                "total_tokens_serialized": 300,
                "episode_clipped": False,
                "termination_reason": "env_done",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "reward_log.jsonl"
            log_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            json_path, md_path = write_episode_summary_outputs(log_path)

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_episodes"], 1)
            self.assertIn("termination_reasons", payload)
            self.assertIn("Episode Run Summary", md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
