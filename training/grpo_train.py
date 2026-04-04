"""GRPO training script using TRL environment_factory with remote OpenEnv."""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from training.config import TrainingRuntimeConfig
from training.openenv_runtime import (
    ReasonBudgetClient,
    to_openenv_base_url,
)


# ---------------------------------------------------------------------------
# Module-level config (set in main() before trainer init)
# ---------------------------------------------------------------------------

ENV_BASE_URL: str = ""
RUNTIME_CFG: TrainingRuntimeConfig | None = None
REWARD_LOG_PATH: str = ""
EPISODE_LOG_COUNT: int = 0
LOG_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are solving math problems under a shared token budget. "
    "Show your reasoning, then give your final answer in \\boxed{}."
)


def _ensure_environment_factory_deps():
    """Fail fast for tool-calling deps required by TRL environment_factory."""
    try:
        import jmespath  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'jmespath'. TRL GRPOTrainer with "
            "environment_factory/tools requires it. Install with "
            "`pip install jmespath` or add it to requirements."
        ) from exc


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


def _write_episode_log(entry: dict):
    """Append one episode-level reward record to reward_logs.jsonl."""
    global EPISODE_LOG_COUNT
    if RUNTIME_CFG is None or not RUNTIME_CFG.log_rewards or not REWARD_LOG_PATH:
        return

    with LOG_LOCK:
        EPISODE_LOG_COUNT += 1
        every_n = max(1, int(RUNTIME_CFG.log_every_n_steps))
        if EPISODE_LOG_COUNT % every_n != 0:
            return
        log_path = Path(REWARD_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Environment class for TRL environment_factory
# ---------------------------------------------------------------------------


class ReasonBudgetToolEnv:
    """TRL environment_factory class wrapping the remote ReasonBudget env.

    The trainer creates one instance per generation, calls reset() at episode
    start, then auto-discovers solve() as a tool the model can invoke.
    """

    def __init__(self):
        self.client = ReasonBudgetClient(base_url=ENV_BASE_URL)
        self.reward = 0.0
        self.done = False
        self._obs: dict | None = None
        self.episode_id = ""
        self.step_logs: list[dict] = []

    def reset(self, **kwargs):
        self.reward = 0.0
        self.done = False
        self.episode_id = uuid4().hex
        self.step_logs = []
        with self.client.sync() as c:
            result = c.reset()
        self._obs = result.observation
        return format_observation_prompt(self._obs)

    def solve(self, response: str):
        """
        Submit your solution to the current math problem.

        Args:
            response: Your full reasoning and final answer in \\boxed{}.

        Returns:
            The next problem observation, or a message that the episode is over.
        """
        if self.done:
            raise ValueError("Episode is over. No more questions.")
        prev_obs = self._obs or {}
        with self.client.sync() as c:
            result = c.step({"response": response})
        self._obs = result.observation
        raw_step_reward = float(result.reward or 0.0)
        step_reward = raw_step_reward
        if RUNTIME_CFG:
            step_reward *= RUNTIME_CFG.alpha
        self.reward += step_reward
        self.done = bool(result.done)

        self.step_logs.append({
            "step_index": len(self.step_logs) + 1,
            "question": prev_obs.get("question", ""),
            "question_summary": prev_obs.get("question_summary", ""),
            "questions_remaining_before": prev_obs.get("questions_remaining"),
            "remaining_budget_before": prev_obs.get("remaining_budget"),
            "tokens_used_before": prev_obs.get("tokens_used"),
            "raw_step_reward": raw_step_reward,
            "scaled_step_reward": step_reward,
            "done_after_step": self.done,
        })

        if self.done:
            _write_episode_log({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "episode_reward": self.reward,
                "num_steps": len(self.step_logs),
                "steps": self.step_logs,
                "final_observation": self._obs,
            })
            return "Episode complete. All questions answered."
        return format_observation_prompt(self._obs)


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer (environment_factory signature)
# ---------------------------------------------------------------------------


def reward_from_env(environments, **kwargs):
    """Read cumulative episode reward from each environment instance."""
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    global ENV_BASE_URL, RUNTIME_CFG, REWARD_LOG_PATH

    parser = argparse.ArgumentParser(description="GRPO training against remote OpenEnv env")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=8192)
    parser.add_argument("--alpha", type=float, default=1.0)
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

    _ensure_environment_factory_deps()

    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    ENV_BASE_URL = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )
    RUNTIME_CFG = TrainingRuntimeConfig(
        alpha=args.alpha,
        log_rewards=not args.no_log_rewards,
        log_every_n_steps=args.log_every_n_steps,
        reward_log_path=args.reward_log_path,
    )
    REWARD_LOG_PATH = RUNTIME_CFG.resolved_reward_log_path(args.output_dir)
    if RUNTIME_CFG.log_rewards:
        print(f"Reward episode logs: {REWARD_LOG_PATH} (every {RUNTIME_CFG.log_every_n_steps} episodes)")

    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Solve the next math problem under budget constraints."},
            ]
        ] * 100
    })

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
        environment_factory=ReasonBudgetToolEnv,
        args=grpo_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
