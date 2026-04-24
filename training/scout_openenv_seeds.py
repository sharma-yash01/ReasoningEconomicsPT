"""Try OpenEnv seeds and group them by reward behavior.

This was useful for finding seeds with mixed rewards before running longer
GRPO jobs. The output manifest can be passed back into grpo_train.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from training.grpo_train import (
    SYSTEM_PROMPT,
    _extract_tool_response,
    _load_tokenizer_for_model,
    format_observation_prompt,
)
from training.openenv_runtime import ReasonBudgetClient, to_openenv_base_url


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _classify_seed(
    *,
    reward_mean: float,
    reward_std: float,
    mixed_std_threshold: float,
    easy_reward_threshold: float,
    hard_reward_threshold: float,
) -> str:
    if reward_std >= mixed_std_threshold:
        return "mixed"
    if reward_mean >= easy_reward_threshold:
        return "easy"
    if reward_mean <= hard_reward_threshold:
        return "hard"
    return "unclear"


def _generate_completions(
    *,
    model,
    tokenizer,
    prompt_messages: list[dict[str, str]],
    device: str,
    num_generations: int,
    max_completion_length: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> list[str]:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_completion_length,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            remove_invalid_values=True,
            renormalize_logits=True,
        )

    completions = []
    for seq in output_ids:
        continuation = seq[prompt_len:]
        completions.append(tokenizer.decode(continuation, skip_special_tokens=True))
    return completions


def _load_model_for_scout(model_name: str, torch_dtype):
    """Try the local cache first, then fall back to a normal Hub load."""
    attempts = [
        {"local_files_only": True},
        {},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load model for {model_name!r}") from last_error


def _grade_seed(
    *,
    base_url: str,
    seed: int,
    completions: list[str],
) -> list[float]:
    rewards: list[float] = []
    for text in completions:
        response = _extract_tool_response(text) or text
        client = ReasonBudgetClient(base_url=base_url)
        try:
            client.connect()
            client.reset(seed=seed)
            result = client.step({"response": response})
            rewards.append(float(result.reward or 0.0))
        finally:
            client.disconnect()
    return rewards


def main():
    parser = argparse.ArgumentParser(
        description="Probe deterministic OpenEnv seeds and bucket them by reward behavior."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--mixed_std_threshold", type=float, default=0.1)
    parser.add_argument("--easy_reward_threshold", type=float, default=0.8)
    parser.add_argument("--hard_reward_threshold", type=float, default=0.0)
    parser.add_argument(
        "--output_path",
        type=str,
        default="runs/openenv_seed_manifest.json",
    )
    args = parser.parse_args()

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    base_url = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )
    device = _resolve_device()

    tokenizer = _load_tokenizer_for_model(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model_for_scout(args.model, dtype_map[args.torch_dtype])
    model.to(device)
    model.eval()

    records = []
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(
        f"Scouting {len(seeds)} OpenEnv seeds with model={args.model} "
        f"on device={device}..."
    )

    for idx, seed in enumerate(seeds, 1):
        client = ReasonBudgetClient(base_url=base_url)
        try:
            client.connect()
            obs = client.reset(seed=seed).observation
        finally:
            client.disconnect()

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation_prompt(obs)},
        ]
        completions = _generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            device=device,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        rewards = _grade_seed(
            base_url=base_url,
            seed=seed,
            completions=completions,
        )
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        bucket = _classify_seed(
            reward_mean=reward_mean,
            reward_std=reward_std,
            mixed_std_threshold=args.mixed_std_threshold,
            easy_reward_threshold=args.easy_reward_threshold,
            hard_reward_threshold=args.hard_reward_threshold,
        )

        metadata = obs.get("metadata", {})
        record = {
            "seed": seed,
            "bucket": bucket,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "rewards": rewards,
            "question_id": metadata.get("question_id"),
            "question_source": metadata.get("question_source"),
            "problem_type": metadata.get("problem_type"),
            "question_preview": obs.get("question", "")[:200],
        }
        records.append(record)
        print(
            f"[{idx}/{len(seeds)}] seed={seed} bucket={bucket} "
            f"mean={reward_mean:.3f} std={reward_std:.3f}"
        )

    summary = {"easy": 0, "mixed": 0, "hard": 0, "unclear": 0}
    for record in records:
        summary[record["bucket"]] += 1

    payload = {
        "model": args.model,
        "env_base_url": base_url,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "thresholds": {
            "mixed_std_threshold": args.mixed_std_threshold,
            "easy_reward_threshold": args.easy_reward_threshold,
            "hard_reward_threshold": args.hard_reward_threshold,
        },
        "summary": summary,
        "records": records,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote seed manifest to {output_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
