"""Unsloth vs Accelerate FSDP sharding compatibility (no heavy imports)."""

from __future__ import annotations

import os


def assert_unsloth_compatible_with_fsdp(use_unsloth: bool) -> None:
    """Raise SystemExit if Unsloth is requested while Accelerate FSDP sharding is enabled."""
    if use_unsloth and os.environ.get("REPT_MODEL_SHARDING", "0").strip() == "1":
        raise SystemExit(
            "--use_unsloth is incompatible with Accelerate FSDP model sharding "
            "(REPT_MODEL_SHARDING=1). Disable sharding or run without --use_unsloth."
        )
