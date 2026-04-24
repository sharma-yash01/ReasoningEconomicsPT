"""GRPO training script for local execution (Apple Silicon / CPU).

Drop-in alternative to grpo_train.py that removes vLLM and uses standard
HF generate() via PyTorch MPS (Apple Silicon) or CPU.

Reward signal: direct math grading on MetaMathQA problems via extract_boxed_answer
+ grade_answer. No env server needed. This bypasses the two approaches that are
unavailable on Mac:
  - environment_factory: requires the model to emit tool-call syntax. Base models
    never learn this format without instruction tuning.
  - rollout_func: TRL 1.0 only wires rollout_func into VLLMGeneration, so it is
    silently ignored when use_vllm=False.

Usage:
    python -m training.grpo_train_local \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --output_dir runs/grpo_local

For Apple Silicon stability with general instruct models, the script defaults to:
  - float32 weights instead of auto half precision
  - non-fused AdamW
  - disabled DataLoader pinned memory
  - generation-side NaN/Inf guards
  - easier GSM-style numeric problems instead of the full mixed MetaMathQA set
  - LoRA adapter training instead of full-parameter updates
"""

from __future__ import annotations

import argparse
import re
import torch

from env.grading import extract_boxed_answer, grade_answer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a math problem solver. "
    "Keep your reasoning brief. End with exactly one final answer in \\boxed{} and stop immediately. "
    "When the answer is numeric, put only the number inside \\boxed{}."
)

_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:!?]+$")
_FINAL_NUMERIC_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
_ANSWER_PATTERNS = (
    re.compile(r"####\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:final\s+answer|the\s+answer|answer|final\s+result|result|solution)\s*(?:is|=|:)\s*(.+)$", re.IGNORECASE),
    re.compile(r"value\s+of\s+[A-Za-z]\w*\s*(?:is|=|:)\s*(.+)$", re.IGNORECASE),
    re.compile(r"(?:therefore|thus|hence|so)[,:]?\s*(.+)$", re.IGNORECASE),
)

MAX_COMPLETION_LENGTH = 0
EOS_TOKEN_ID: int | None = None
FORMAT_REWARD_BONUS = 0.05
TRUNCATION_PENALTY = 0.05


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def _clean_answer_candidate(candidate: str) -> str | None:
    """Normalize an extracted answer span before grading."""
    candidate = candidate.strip()
    if not candidate:
        return None

    boxed = extract_boxed_answer(candidate)
    if boxed:
        return boxed

    candidate = candidate.strip().strip("$").strip()
    candidate = candidate.lstrip(":= ").strip()
    candidate = _TRAILING_PUNCT_RE.sub("", candidate)
    if not candidate:
        return None

    return candidate.replace(",", "")


def _extract_final_answer(text: str) -> str | None:
    """Extract the final answer from model output.

    Tries in order:
      1. \\boxed{...}: preferred format
      2. Answer-signaling phrases in the last 5 lines ("The answer is: X",
         "#### X", "value of x is X"): catches prose-form answers
      3. Last numeric expression in the full text: coarse fallback
    """
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    for line in reversed(lines[-5:]):
        for pattern in _ANSWER_PATTERNS:
            match = pattern.search(line)
            if match:
                cleaned = _clean_answer_candidate(match.group(1))
                if cleaned is not None:
                    return cleaned

    numbers = _FINAL_NUMERIC_RE.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def _is_simple_numeric_answer(answer: str) -> bool:
    """Return True for plain numeric targets such as 12, -3.5, or 7/8."""
    return _FINAL_NUMERIC_RE.fullmatch(answer) is not None


def math_reward(prompts, completions, ground_truth, **kwargs):
    """Grade each completion against the ground-truth answer.

    Returns:
      - 1.0 for a correct final answer
      - a small bonus for a parseable-but-wrong final answer
      - a small penalty for likely-truncated wrong completions

    This keeps correctness as the main signal, while making the reward slightly
    denser and nudging the model toward concise, parseable outputs.

    Args:
        prompts: list of prompt messages (unused, kept for TRL signature).
        completions: list of model outputs. Each is either a string or a
            list of chat-format dicts like [{"role": "assistant", ...}].
        ground_truth: list of ground-truth answer strings from the dataset.
    """
    completion_ids = kwargs.get("completion_ids")
    log_metric = kwargs.get("log_metric")

    rewards = []
    parseable_count = 0
    truncated_count = 0
    correct_count = 0
    for i, (completion, gt) in enumerate(zip(completions, ground_truth)):
        # Unwrap chat-format dict if present
        if isinstance(completion, list) and isinstance(completion[0], dict):
            text = completion[0].get("content", "")
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)

        predicted = _extract_final_answer(text)
        is_correct = grade_answer(predicted, gt)
        parseable = predicted is not None
        ids = completion_ids[i] if completion_ids is not None else None
        was_truncated = (
            ids is not None
            and MAX_COMPLETION_LENGTH > 0
            and len(ids) >= MAX_COMPLETION_LENGTH
            and (EOS_TOKEN_ID is None or ids[-1] != EOS_TOKEN_ID)
        )

        reward = 1.0 if is_correct else 0.0
        if parseable and not is_correct:
            reward += FORMAT_REWARD_BONUS
        if was_truncated and not is_correct:
            reward -= TRUNCATION_PENALTY

        rewards.append(reward)
        parseable_count += int(parseable)
        truncated_count += int(was_truncated)
        correct_count += int(is_correct)

    if log_metric is not None and rewards:
        denom = float(len(rewards))
        log_metric("reward/parseable_rate", parseable_count / denom)
        log_metric("reward/truncated_rate", truncated_count / denom)
        log_metric("reward/correct_rate", correct_count / denom)

    return rewards


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training on MetaMathQA, local/MPS mode (no vLLM, no env server)"
    )
    # Default to a general instruct model: it follows answer-format directions
    # better than Qwen3.5-0.8B base, and is a cleaner drop-in than Qwen3's
    # default thinking-mode behavior for the current script.
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2)
    # Shorter completions reduce clipping and keep local GRPO stable on MPS.
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="runs/grpo_local")
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    # Number of MetaMathQA problems to train on (dataset is 395K problems)
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    # Keep some diversity, but avoid the very high-entropy 1.5 setting that
    # produced long clipped outputs and unstable sampling.
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    # MPS is more stable with the unfused optimizer and tighter clipping.
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--problem_family",
        type=str,
        default="gsm",
        choices=["gsm", "math", "all"],
    )
    parser.add_argument("--numeric_only", action="store_true", default=True)
    parser.add_argument("--allow_non_numeric", action="store_false", dest="numeric_only")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--format_reward_bonus", type=float, default=0.05)
    parser.add_argument("--truncation_penalty", type=float, default=0.05)
    args = parser.parse_args()

    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    global MAX_COMPLETION_LENGTH, EOS_TOKEN_ID, FORMAT_REWARD_BONUS, TRUNCATION_PENALTY

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # ------------------------------------------------------------------
    # Build dataset: load real math problems + extract ground-truth answers
    # ------------------------------------------------------------------
    print(f"Loading {args.n_problems} problems from MetaMathQA...")
    raw = load_dataset("meta-math/MetaMathQA", split="train")
    raw = raw.shuffle(seed=args.seed)

    prompts, ground_truths = [], []
    for row in raw:
        row_type = row.get("type", "")
        if args.problem_family == "gsm" and not row_type.startswith("GSM"):
            continue
        if args.problem_family == "math" and not row_type.startswith("MATH"):
            continue

        # MetaMathQA usually ends with either \boxed{answer}, "The answer is: X", or "#### X".
        gt = _extract_final_answer(row["response"])
        if gt is None:
            continue
        if args.numeric_only and not _is_simple_numeric_answer(gt):
            continue
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["query"]},
        ])
        ground_truths.append(gt)
        if len(prompts) >= args.n_problems:
            break

    if len(prompts) < args.n_problems:
        raise ValueError(
            f"Only found {len(prompts)} usable problems for family={args.problem_family!r}, "
            f"numeric_only={args.numeric_only}."
        )

    # ground_truth column is forwarded to math_reward by GRPOTrainer
    dataset = Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})

    # ------------------------------------------------------------------
    # Configure trainer
    # ------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        # No vLLM. Use standard HF generate() via MPS (Apple Silicon) or CPU.
        use_vllm=False,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        logging_steps=1,
        save_strategy="epoch",
        dataloader_pin_memory=False,
        # Recompute activations during backward. This saves about 40% memory on MPS.
        gradient_checkpointing=True,
        # Sampling settings tuned for local instruct-model runs: enough variation
        # for GRPO groups, but less clipping/instability than temperature=1.5.
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        mask_truncated_completions=True,
        model_init_kwargs={"torch_dtype": dtype_map[args.torch_dtype]},
        generation_kwargs={
            # Guard against invalid logits during sampling on MPS/general-model runs.
            "remove_invalid_values": True,
            "renormalize_logits": True,
        },
    )

    # Load the tokenizer explicitly instead of AutoProcessor so text-only GRPO
    # stays stable across Qwen variants.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_COMPLETION_LENGTH = args.max_completion_length
    EOS_TOKEN_ID = tokenizer.eos_token_id
    FORMAT_REWARD_BONUS = args.format_reward_bonus
    TRUNCATION_PENALTY = args.truncation_penalty

    peft_config = None
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType
        except ImportError as exc:
            raise ImportError(
                "LoRA is enabled by default for local stability, but `peft` is not installed. "
                "Install it with `.venv-local/bin/python -m pip install peft` or rerun with `--no_lora`."
            ) from exc

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=math_reward,
        train_dataset=dataset,
        args=grpo_config,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
