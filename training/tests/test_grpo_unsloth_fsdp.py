"""FSDP vs Unsloth guard and bundled profile coverage for Gemma 4."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from training.unsloth_fsdp import assert_unsloth_compatible_with_fsdp
from training.model_profiles import load_profiles


def _transformers_version_tuple() -> tuple[int, int]:
    try:
        import transformers

        parts = transformers.__version__.split(".")[:2]
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


class TestUnslothFSDPGuard(unittest.TestCase):
    def test_ok_when_unsloth_off(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            assert_unsloth_compatible_with_fsdp(False)

    def test_ok_when_sharding_off(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "0"}, clear=False):
            assert_unsloth_compatible_with_fsdp(True)

    def test_exit_when_both(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            with self.assertRaises(SystemExit):
                assert_unsloth_compatible_with_fsdp(True)


class TestBundledProfilesGemma4(unittest.TestCase):
    def test_gemma4_hub_id_matches_profile(self):
        path = Path(__file__).resolve().parent.parent / "model_profiles.json"
        reg = load_profiles(path)
        r = reg.resolve("google/gemma-4-26B-A4B-it")
        self.assertEqual(r.output_parser, "gemma4_think")
        self.assertTrue(r.grading_use_visible_only)
        self.assertEqual(r.chat_template_kwargs.get("thinking_mode"), True)


class TestUnslothOptionalImport(unittest.TestCase):
    def test_unsloth_symbols_when_installed(self):
        try:
            from unsloth import FastLanguageModel, PatchFastRL
        except ImportError:
            self.skipTest("unsloth not installed (optional; pip install unsloth)")
        self.assertIsNotNone(FastLanguageModel)
        self.assertIsNotNone(PatchFastRL)


class TestGemma4HubConfig(unittest.TestCase):
    @unittest.skipIf(
        _transformers_version_tuple() < (5, 5),
        "transformers>=5.5 required for gemma4 config",
    )
    def test_auto_config_gemma4_model_type(self):
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained("google/gemma-4-26B-A4B-it")
        self.assertEqual(cfg.model_type, "gemma4")


if __name__ == "__main__":
    unittest.main()
