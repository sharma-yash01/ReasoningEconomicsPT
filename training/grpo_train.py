"""GRPO training script using remote OpenEnv client via base URL."""

import argparse
import json
from pathlib import Path
from typing import Any

from training.config import TrainingRuntimeConfig
from training.openenv_runtime import (
    ReasonBudgetClient,
    resolve_budget_mode_from_observation,
    to_openenv_base_url,
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are solving math problems under a shared token budget. "
    "Show your reasoning, then give your final answer in \\boxed{}."
)


def format_observation_prompt(obs: dict):
    """Format an environment observation into a natural language prompt for the LLM."""
    history = obs.get("history", [])
    if history:
        entries = []
        for i, h in enumerate(history, 1):
            status = "correct" if h.get("was_correct") else "wrong"
            tokens = h.get("tokens_used", "?")
            summary = h.get("question_summary", "")
            entries.append(f"  Q{i}: {summary}... [{tokens} tokens, {status}]")
        history_lines = "\n".join(entries)
    else:
        history_lines = "  (none yet)"

    return (
        f"Remaining budget: {int(obs['remaining_budget'])} tokens\n"
        f"Questions remaining: {obs['questions_remaining']} (including this one)\n"
        f"Budget per remaining question: {obs['budget_per_remaining']:.0f} tokens\n"
        f"Your accuracy so far: {obs['accuracy_so_far']:.0%}\n"
        f"\nPrevious questions:\n{history_lines}\n"
        f"\nCurrent question:\n{obs['question']}\n"
        f"\nSolve this problem. Show your reasoning, then give your final answer in \\boxed{{}}."
    )


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer
# ---------------------------------------------------------------------------


def reward_from_env(completions: list, **kwargs: Any):
    """Extract environment rewards for GRPOTrainer's reward_funcs interface."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(r) for r in env_rewards]
    return [0.0] * len(completions or [])


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def build_rollout_func(*, env_base_url: str, runtime_cfg: TrainingRuntimeConfig, output_dir: str):
    """Build rollout function bound to target OpenEnv endpoint."""
    reward_log_path = runtime_cfg.resolved_reward_log_path(output_dir)
    reward_log_file = Path(reward_log_path)
    reward_log_file.parent.mkdir(parents=True, exist_ok=True)

    if abs(runtime_cfg.beta) > 1e-12:
        print(
            "Note: beta is configured but reward decomposition is unavailable. "
            "Training currently uses total incoming reward signal only."
        )

    def rollout_func(prompts: list[str], trainer: Any):
        """Core rollout loop against remote OpenEnv ReasonBudget environment.

        For each prompt in the batch, run a full episode:
        1. Reset env via OpenEnv client
        2. For each step, format observation -> generate completion -> step env
        3. Collect prompt_ids, completion_ids, logprobs, and rewards
        """
        from trl.trainer.grpo_trainer import generate_rollout_completions

        tokenizer = trainer.processing_class

        all_prompt_ids: list = []
        all_completion_ids: list = []
        all_logprobs: list = []
        all_rewards: list[float] = []

        with ReasonBudgetClient(base_url=env_base_url).sync() as env_client:
            for episode_idx, _prompt in enumerate(prompts):
                result = env_client.reset()
                obs = result.observation
                done = bool(result.done)
                step_idx = 0
                episode_reward = 0.0

                while not done:
                    user_prompt = format_observation_prompt(obs)

                    budget_mode = resolve_budget_mode_from_observation(
                        obs,
                        default_mode=runtime_cfg.normalized_default_mode(),
                        strict=runtime_cfg.strict_budget_mode_metadata,
                    )
                    remaining_budget = int(obs.get("remaining_budget", 0))
                    if budget_mode == "hard":
                        max_new_tokens = min(
                            runtime_cfg.max_tokens_per_step,
                            max(0, remaining_budget),
                        )
                        max_new_tokens = max(
                            max_new_tokens,
                            runtime_cfg.min_tokens_per_step,
                        )
                    else:
                        max_new_tokens = runtime_cfg.max_tokens_per_step

                    outputs = generate_rollout_completions(
                        trainer,
                        [user_prompt],
                        max_new_tokens=max_new_tokens,
                    )[0]

                    completion_text = tokenizer.decode(
                        outputs["completion_ids"], skip_special_tokens=True
                    )

                    result = env_client.step({"response": completion_text})
                    obs = result.observation
                    done = bool(result.done)
                    total_signal = float(result.reward or 0.0)
                    weighted_signal = runtime_cfg.alpha * total_signal
                    episode_reward += weighted_signal

                    if runtime_cfg.log_rewards and step_idx % max(1, runtime_cfg.log_every_n_steps) == 0:
                        print(
                            f"[reward] ep={episode_idx} step={step_idx} mode={budget_mode} "
                            f"raw_total={total_signal:.6f} weighted={weighted_signal:.6f} done={done}"
                        )
                        with reward_log_file.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "episode_idx": episode_idx,
                                        "step_idx": step_idx,
                                        "budget_mode": budget_mode,
                                        "raw_total_reward": total_signal,
                                        "weighted_reward": weighted_signal,
                                        "done": done,
                                    }
                                )
                                + "\n"
                            )

                    all_prompt_ids.append(outputs["prompt_ids"])
                    all_completion_ids.append(outputs["completion_ids"])
                    all_logprobs.append(outputs["logprobs"])
                    step_idx += 1

                all_rewards.append(episode_reward)
                if runtime_cfg.log_rewards:
                    with reward_log_file.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "episode_idx": episode_idx,
                                    "episode_weighted_reward": episode_reward,
                                    "event": "episode_end",
                                }
                            )
                            + "\n"
                        )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_rewards,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GRPO training against remote OpenEnv env")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_tokens_per_step", type=int, default=2048)
    parser.add_argument("--min_tokens_per_step", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--default_budget_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--strict_budget_mode_metadata", action="store_true")
    parser.add_argument("--no_log_rewards", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--reward_log_path", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument("--output_dir", type=str, default="runs/grpo_train")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    args = parser.parse_args()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    env_base_url = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )
    runtime_cfg = TrainingRuntimeConfig(
        alpha=args.alpha,
        beta=args.beta,
        max_tokens_per_step=args.max_tokens_per_step,
        min_tokens_per_step=args.min_tokens_per_step,
        default_budget_mode=args.default_budget_mode,
        strict_budget_mode_metadata=args.strict_budget_mode_metadata,
        log_rewards=not args.no_log_rewards,
        log_every_n_steps=args.log_every_n_steps,
        reward_log_path=args.reward_log_path,
    )

    # Placeholder dataset -- rollout_func drives the actual episode data
    dataset = Dataset.from_dict({"prompt": ["placeholder"] * 100})

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_strategy="epoch",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=build_rollout_func(
            env_base_url=env_base_url,
            runtime_cfg=runtime_cfg,
            output_dir=args.output_dir,
        ),
        args=grpo_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
