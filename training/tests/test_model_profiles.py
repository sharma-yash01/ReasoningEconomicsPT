"""Unit tests for model_profiles registry and Qwen think parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.model_profiles import (
    merge_chat_template_kwargs_for_reasoning_mode,
    parse_completion,
    profile_lookup_model_id,
)
from training.model_profiles import load_profiles

_THINK_O = "<" + "think" + ">"
_THINK_C = "</" + "think" + ">"


class TestProfileLookupModelId(unittest.TestCase):
    def test_hub_id_passthrough(self):
        self.assertEqual(
            profile_lookup_model_id(model_arg="Qwen/Qwen3-8B", env_tokenizer_name=None),
            "Qwen/Qwen3-8B",
        )

    def test_local_uses_env_tokenizer(self):
        self.assertEqual(
            profile_lookup_model_id(
                model_arg="/lambda/nfs/models/Qwen_Qwen3-8B",
                env_tokenizer_name="Qwen/Qwen3-8B",
            ),
            "Qwen/Qwen3-8B",
        )


class TestMergeReasoningMode(unittest.TestCase):
    def test_auto_unchanged(self):
        base = {"enable_thinking": True, "foo": 1}
        self.assertEqual(merge_chat_template_kwargs_for_reasoning_mode(base, reasoning_mode="auto"), base)

    def test_on_off(self):
        self.assertTrue(
            merge_chat_template_kwargs_for_reasoning_mode({}, reasoning_mode="on")["enable_thinking"]
        )
        self.assertFalse(
            merge_chat_template_kwargs_for_reasoning_mode({"enable_thinking": True}, reasoning_mode="off")[
                "enable_thinking"
            ]
        )


class TestRegistryResolve(unittest.TestCase):
    def test_longest_prefix_wins(self):
        data = {
            "version": 1,
            "default": {
                "chat_template_kwargs": {},
                "output_parser": None,
                "think_tag_open": None,
                "think_tag_close": None,
                "grading_use_visible_only": False,
            },
            "profiles": [
                {
                    "match_prefixes": ["Qwen/Qwen3"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "output_parser": None,
                    "grading_use_visible_only": False,
                },
                {
                    "match_prefixes": ["Qwen/Qwen3.5"],
                    "chat_template_kwargs": {"enable_thinking": True},
                    "output_parser": "qwen3_think",
                    "grading_use_visible_only": True,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            p = Path(f.name)
        try:
            reg = load_profiles(p)
            r = reg.resolve("Qwen/Qwen3.5-9B")
            self.assertTrue(r.chat_template_kwargs.get("enable_thinking"))
            self.assertEqual(r.output_parser, "qwen3_think")
            r2 = reg.resolve("Qwen/Qwen3-8B")
            self.assertFalse(r2.chat_template_kwargs.get("enable_thinking"))
        finally:
            p.unlink(missing_ok=True)

    def test_exact_match(self):
        data = {
            "version": 1,
            "default": {
                "chat_template_kwargs": {},
                "output_parser": None,
                "think_tag_open": None,
                "think_tag_close": None,
                "grading_use_visible_only": False,
            },
            "profiles": [
                {
                    "match_exact": ["Special/Model"],
                    "match_prefixes": [],
                    "chat_template_kwargs": {"x": 1},
                    "output_parser": None,
                    "grading_use_visible_only": False,
                },
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            p = Path(f.name)
        try:
            reg = load_profiles(p)
            self.assertEqual(reg.resolve("Special/Model").chat_template_kwargs.get("x"), 1)
            self.assertNotIn("x", reg.resolve("Special/ModelX").chat_template_kwargs)
        finally:
            p.unlink(missing_ok=True)


class TestParseCompletion(unittest.TestCase):
    def test_no_parser(self):
        p = parse_completion("hello", None)
        self.assertEqual(p.full, "hello")
        self.assertEqual(p.visible, "hello")
        self.assertEqual(p.reasoning, "")

    def test_qwen_with_block(self):
        text = "<think>\nstep\n</think>\n\n\\boxed{42}"
        p = parse_completion(text, "qwen3_think", think_tag_open="<think>", think_tag_close="</think>")
        self.assertIn("step", p.reasoning)
        self.assertIn("42", p.visible)
        self.assertEqual(p.full, text)

    def test_qwen_no_open_tag(self):
        text = "plain \\boxed{1}"
        p = parse_completion(text, "qwen3_think", think_tag_open="<think>", think_tag_close="</think>")
        self.assertEqual(p.reasoning, "")
        self.assertEqual(p.visible, text)

    def test_qwen_unclosed(self):
        text = "<think>\nonly"
        p = parse_completion(text, "qwen3_think", think_tag_open="<think>", think_tag_close="</think>")
        self.assertEqual(p.reasoning, "only")
        self.assertEqual(p.visible, "")


if __name__ == "__main__":
    unittest.main()
