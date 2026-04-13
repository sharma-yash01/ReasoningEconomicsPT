"""Detect Accelerate FSDP model sharding (launcher env) for Unsloth + GRPO."""

from __future__ import annotations

import os
import warnings


def is_fsdp_model_sharding_env() -> bool:
    """True when Lambda / scripts set ``REPT_MODEL_SHARDING=1`` (``accelerate launch`` + FSDP yaml)."""
    return os.environ.get("REPT_MODEL_SHARDING", "0").strip() == "1"


def warn_unsloth_fsdp_experimental(*, use_unsloth: bool, load_in_4bit: bool) -> None:
    """Log that Unsloth + FSDP is experimental; optional QLoRA + FSDP caveats."""
    if not use_unsloth or not is_fsdp_model_sharding_env():
        return
    warnings.warn(
        "Unsloth + REPT_MODEL_SHARDING=1 (FSDP): experimental. Use `accelerate launch` with your "
        "model-sharding yaml; GRPOTrainer/Accelerate should wrap the policy. "
        "If init or weight-sync fails, try without --load_in_4bit or adjust fsdp_config.",
        UserWarning,
        stacklevel=2,
    )
    if load_in_4bit:
        warnings.warn(
            "Unsloth + FSDP + --load_in_4bit: QLoRA under FSDP can be stack-sensitive; "
            "fallback to bf16/16-bit LoRA if you see bitsandbytes or shard errors.",
            UserWarning,
            stacklevel=2,
        )
