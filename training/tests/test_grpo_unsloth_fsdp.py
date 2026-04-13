"""FSDP env detection for Unsloth + bundled Gemma 4 profile tests."""

from __future__ import annotations

import os
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from training.unsloth_fsdp import is_fsdp_model_sharding_env, warn_unsloth_fsdp_experimental
from training.model_profiles import load_profiles


class TestFsdpShardingEnv(unittest.TestCase):
    def test_false_when_zero(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "0"}, clear=False):
            self.assertFalse(is_fsdp_model_sharding_env())

    def test_true_when_one(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            self.assertTrue(is_fsdp_model_sharding_env())


class TestWarnUnslothFsdp(unittest.TestCase):
    def test_no_warn_when_not_unsloth(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_unsloth_fsdp_experimental(use_unsloth=False, load_in_4bit=False)
                self.assertEqual(len(w), 0)

    def test_warns_when_both(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_unsloth_fsdp_experimental(use_unsloth=True, load_in_4bit=False)
                self.assertTrue(any("experimental" in str(x.message).lower() for x in w))

    def test_extra_warn_4bit(self):
        with patch.dict(os.environ, {"REPT_MODEL_SHARDING": "1"}, clear=False):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                warn_unsloth_fsdp_experimental(use_unsloth=True, load_in_4bit=True)
                msgs = " ".join(str(x.message).lower() for x in w)
                self.assertIn("experimental", msgs)
                self.assertIn("load_in_4bit", msgs)


class TestBundledProfilesGemma4(unittest.TestCase):
    def test_gemma4_hub_id_matches_profile(self):
        path = Path(__file__).resolve().parent.parent / "model_profiles.json"
        reg = load_profiles(path)
        r = reg.resolve("google/gemma-4-26B-A4B-it")
        self.assertEqual(r.output_parser, "gemma4_think")
        self.assertTrue(r.grading_use_visible_only)
        self.assertEqual(r.chat_template_kwargs.get("thinking_mode"), True)


def _transformers_version_tuple() -> tuple[int, int]:
    try:
        import transformers

        parts = transformers.__version__.split(".")[:2]
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0


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
