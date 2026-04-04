"""Run all baselines over N episodes against remote env; collect metrics."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval.baselines import (
    APIChatBaseline,
    DifficultyOracleBaseline,
    GreedyMaxBaseline,
    LocalVLLMBaseline,
    UniformBaseline,
)
from training.openenv_runtime import ReasonBudgetClient, to_openenv_base_url


def _select_response(
    obs: dict,
    baseline: Any,
    max_new_tokens: int | None = None,
):
    ptype = obs.get("metadata", {}).get("problem_type")
    return baseline.select_action(
        obs,
        problem_type=ptype,
        max_new_tokens=max_new_tokens,
    )


def _parse_csv_names(value: str | None):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_baselines(
    args: argparse.Namespace,
    *,
    min_tokens: int,
    max_tokens: int,
):
    baselines: dict[str, Any] = {
        "uniform": UniformBaseline(min_tokens, max_tokens),
        "greedy_max": GreedyMaxBaseline(min_tokens, max_tokens),
        "oracle": DifficultyOracleBaseline(min_tokens, max_tokens),
    }

    requested = _parse_csv_names(args.baselines)
    needs_llm = args.include_llm or any(name.startswith("llm_") for name in requested)
    if needs_llm:
        baselines["llm_api"] = APIChatBaseline(
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )
        baselines["llm_local"] = LocalVLLMBaseline(
            timeout_s=args.llm_timeout_s,
            max_retries=args.llm_max_retries,
            temperature=args.llm_temperature,
        )

    if not requested:
        return baselines

    unknown = [name for name in requested if name not in baselines]
    if unknown:
        available = ", ".join(sorted(baselines.keys()))
        raise ValueError(
            f"Unknown baselines: {unknown}. Available choices: {available}"
        )

    return {name: baselines[name] for name in requested}


def evaluate_baseline(
    env_client,
    baseline,
    n_episodes: int,
    seed: int,
    llm_max_new_tokens: int | None = None,
    env_tokenizer_name: str | None = None,
):
    results = []
    for ep in range(n_episodes):
        reset_kw: dict = {"seed": seed + ep}
        if env_tokenizer_name:
            reset_kw["tokenizer_name"] = env_tokenizer_name
        result = env_client.reset(**reset_kw)
        obs = result.observation
        total_reward = 0.0
        tokens_per_step: list[int] = []
        while not obs.get("done", False):
            response = _select_response(
                obs,
                baseline,
                max_new_tokens=llm_max_new_tokens,
            )
            step_payload: dict = {"response": response}
            if env_tokenizer_name:
                step_payload["metadata"] = {"tokenizer_name": env_tokenizer_name}
            result = env_client.step(step_payload)
            obs = result.observation
            total_reward += float(result.reward or 0.0)
            history = obs.get("history", [])
            if history:
                tokens_per_step.append(history[-1].get("tokens_used", 0))
        state = env_client.state()
        spent = int(state.get("spent_budget", 0))
        budget = int(state.get("total_budget", 0))
        budget_util = (spent / budget) if budget > 0 else 0.0
        overspend_tokens = max(0, spent - budget)
        results.append({
            "total_reward": total_reward,
            "total_correct": state.get("total_correct", 0),
            "questions_answered": state.get("questions_answered", 0),
            "accuracy": (
                state.get("total_correct", 0) / state.get("questions_answered", 1)
                if state.get("questions_answered", 0)
                else 0
            ),
            "budget_utilization": budget_util,
            "budget_utilization_clamped": min(1.0, budget_util),
            "overspend_tokens": overspend_tokens,
            "went_over_budget": overspend_tokens > 0,
            "tokens_per_step": tokens_per_step,
            "mean_tokens_per_question": (
                float(np.mean(tokens_per_step)) if tokens_per_step else 0.0
            ),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument(
        "--include_llm",
        action="store_true",
        help="Include llm_api and llm_local baselines (requires env vars).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="",
        help="Comma-separated baseline names to run (default: all enabled).",
    )
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--llm_timeout_s", type=float, default=30.0)
    parser.add_argument("--llm_max_retries", type=int, default=2)
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument(
        "--env_tokenizer_name",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model id passed to the remote env on reset/step "
            "(same as GRPO training --env_tokenizer_name)."
        ),
    )
    args = parser.parse_args()

    env_base_url = to_openenv_base_url(
        env_base_url=args.env_base_url,
        space_url=args.space_url,
    )

    with ReasonBudgetClient(base_url=env_base_url).sync() as env_client:
        # Probe config from the first reset observation metadata
        probe_kw: dict = {"seed": args.seed}
        if args.env_tokenizer_name:
            probe_kw["tokenizer_name"] = args.env_tokenizer_name.strip()
        probe = env_client.reset(**probe_kw)
        probe_meta = probe.observation.get("metadata", {})
        min_tokens = probe_meta.get("min_tokens", 10)
        max_tokens = probe_meta.get("max_tokens", 800)

        all_results: dict[str, list[dict]] = {}
        selected_baselines = _build_baselines(
            args,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )

        for name, baseline in selected_baselines.items():
            res = evaluate_baseline(
                env_client,
                baseline,
                args.n_episodes,
                args.seed,
                llm_max_new_tokens=args.llm_max_new_tokens,
                env_tokenizer_name=(
                    args.env_tokenizer_name.strip() if args.env_tokenizer_name else None
                ),
            )
            all_results[name] = res

    summary = {}
    for agent, runs in all_results.items():
        accs = [r["accuracy"] for r in runs]
        rewards = [r["total_reward"] for r in runs]
        budget_utils = [r["budget_utilization"] for r in runs]
        budget_utils_clamped = [r.get("budget_utilization_clamped", 0.0) for r in runs]
        overspends = [r.get("overspend_tokens", 0) for r in runs]
        went_over = [1.0 if r.get("went_over_budget", False) else 0.0 for r in runs]
        mean_tpq = [r["mean_tokens_per_question"] for r in runs]
        summary[agent] = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "budget_utilization_mean": float(np.mean(budget_utils)),
            "budget_utilization_clamped_mean": float(np.mean(budget_utils_clamped)),
            "overspend_tokens_mean": float(np.mean(overspends)),
            "episodes_over_budget_rate": float(np.mean(went_over)),
            "mean_tokens_per_question": float(np.mean(mean_tpq)),
        }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_compact = {}
    for agent, runs in all_results.items():
        raw_compact[agent] = [
            {k: v for k, v in r.items() if k != "tokens_per_step"} for r in runs
        ]
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "raw": raw_compact}, f, indent=1)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
