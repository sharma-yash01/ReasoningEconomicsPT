"""GRPO training with TRL rollout_func and remote OpenEnv (explicit env stepping)."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import warnings

import torch
import torch.distributed as dist
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from training.config import TrainingRuntimeConfig
from training.unsloth_fsdp import is_fsdp_model_sharding_env, warn_unsloth_fsdp_experimental
from training.model_profiles import (
    ParsedCompletion,
    ResolvedProfile,
    load_profiles,
    merge_chat_template_kwargs_for_reasoning_mode,
    parse_completion,
    profile_lookup_model_id,
)
from training.openenv_runtime import (
    ReasonBudgetClient,
    resolve_budget_mode_from_observation,
    to_openenv_base_url,
)

if TYPE_CHECKING:
    from trl import GRPOTrainer


# ---------------------------------------------------------------------------
# Module-level config (set in main() before trainer init)
# ---------------------------------------------------------------------------

ENV_BASE_URL: str = ""
RUNTIME_CFG: TrainingRuntimeConfig | None = None
REWARD_LOG_PATH: str = ""
EPISODE_LOG_COUNT: int = 0
LOG_LOCK = threading.Lock()

LOG_PREVIEW_CHARS = 2000

# TRL vLLM server mode calls NCCL collectives inside each generate(); keep call counts
# identical across ranks (see impl-context/dist-train-issue-hung-gpu.md).
DIST_SERVER_GENERATES_PER_EPISODE = 8


def _truncate_for_log(s: str, max_len: int = LOG_PREVIEW_CHARS) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…[truncated]"


def _parse_completion_for_profile(text: str, profile: ResolvedProfile | None) -> ParsedCompletion:
    if profile and profile.output_parser:
        return parse_completion(
            text,
            profile.output_parser,
            think_tag_open=profile.think_tag_open,
            think_tag_close=profile.think_tag_close,
        )
    return parse_completion(text, None)


def _build_env_step_metadata(
    tokenizer_name: str,
    profile: ResolvedProfile | None,
    parsed: ParsedCompletion,
) -> dict[str, Any]:
    md: dict[str, Any] = {"tokenizer_name": tokenizer_name}
    if profile and profile.grading_use_visible_only and parsed.visible.strip():
        md["grading_response"] = parsed.visible
    return md


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


def resolve_env_tokenizer_name(
    tok,
    trainer: "GRPOTrainer",
    override: str | None,
    *,
    fallback_model_id: str | None = None,
) -> str:
    """Hugging Face tokenizer / model id to send to the remote env (``AutoTokenizer.from_pretrained``)."""
    if override and str(override).strip():
        return str(override).strip()
    name = getattr(tok, "name_or_path", None)
    if isinstance(name, str) and name.strip():
        s = name.strip()
        p = Path(s)
        if p.is_absolute():
            warnings.warn(
                f"Tokenizer name_or_path looks like a local path ({s!r}); the remote env "
                "cannot load it. Pass --env_tokenizer_name with a Hugging Face model id.",
                UserWarning,
                stacklevel=2,
            )
        return s
    model = getattr(trainer, "model", None)
    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            mid = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
            if isinstance(mid, str) and mid.strip():
                return mid.strip()
    if fallback_model_id and str(fallback_model_id).strip():
        return str(fallback_model_id).strip()
    raise ValueError(
        "Could not resolve a Hugging Face model id for the remote env tokenizer. "
        "Pass --env_tokenizer_name explicitly."
    )


# ---------------------------------------------------------------------------
# Episode session (shared reset / step logic)
# ---------------------------------------------------------------------------


class EpisodeSession:
    """One remote episode: reset + step with response text (same contract as former ``solve``).

    OpenEnv binds one server env per WebSocket; ``reset`` and every ``step`` must use the same
    connection. Use ``with EpisodeSession(...) as session:`` for the full episode.
    """

    def __init__(
        self,
        base_url: str,
        *,
        tokenizer_name: str,
        total_budget: int | None = None,
    ):
        self.client = ReasonBudgetClient(base_url=base_url)
        self.tokenizer_name = tokenizer_name
        self.total_budget_override = total_budget
        self.reward = 0.0
        self.done = False
        self._obs: dict | None = None
        self.episode_id = ""
        self.step_logs: list[dict] = []
        self._env: Any = None
        self._conn_cm: Any = None

    def __enter__(self) -> EpisodeSession:
        sync_maker = getattr(self.client, "sync", None)
        if callable(sync_maker):
            self._conn_cm = sync_maker()
            self._env = self._conn_cm.__enter__()
        else:
            self._conn_cm = None
            self.client.__enter__()
            self._env = self.client
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:
        try:
            if self._conn_cm is not None:
                return self._conn_cm.__exit__(exc_type, exc_val, exc_tb)
            return self.client.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._env = None
            self._conn_cm = None

    def reset_episode(self) -> dict:
        if self._env is None:
            raise RuntimeError(
                "EpisodeSession must be used as a context manager "
                "(with EpisodeSession(...) as session: ...)."
            )
        self.reward = 0.0
        self.done = False
        self.episode_id = uuid4().hex
        self.step_logs = []
        reset_kwargs: dict[str, Any] = {"tokenizer_name": self.tokenizer_name}
        if self.total_budget_override is not None:
            reset_kwargs["total_budget"] = self.total_budget_override
        result = self._env.reset(**reset_kwargs)
        self._obs = result.observation
        return self._obs

    def apply_response(
        self,
        response: str,
        step_metadata: dict[str, Any] | None = None,
        log_extras: dict[str, Any] | None = None,
    ):
        """Step env with model text; updates ``reward``, ``done``, ``_obs``, ``step_logs``."""
        if self._env is None:
            raise RuntimeError(
                "EpisodeSession must be used as a context manager "
                "(with EpisodeSession(...) as session: ...)."
            )
        if self.done:
            raise ValueError("Episode is over. No more questions.")
        prev_obs = self._obs or {}
        md: dict[str, Any] = {"tokenizer_name": self.tokenizer_name}
        if step_metadata:
            for k, v in step_metadata.items():
                if v is not None:
                    md[k] = v
        result = self._env.step({
            "response": response,
            "metadata": md,
        })
        self._obs = result.observation
        raw_step_reward = float(result.reward or 0.0)
        step_reward = raw_step_reward
        if RUNTIME_CFG:
            step_reward *= RUNTIME_CFG.alpha
        self.reward += step_reward
        self.done = bool(result.done)

        entry: dict[str, Any] = {
            "step_index": len(self.step_logs) + 1,
            "question": prev_obs.get("question", ""),
            "question_summary": prev_obs.get("question_summary", ""),
            "questions_remaining_before": prev_obs.get("questions_remaining"),
            "remaining_budget_before": prev_obs.get("remaining_budget"),
            "tokens_used_before": prev_obs.get("tokens_used"),
            "model_response": response,
            "raw_step_reward": raw_step_reward,
            "scaled_step_reward": step_reward,
            "done_after_step": self.done,
        }
        if log_extras:
            entry.update(log_extras)
        self.step_logs.append(entry)

        if self.done:
            _write_episode_log({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "episode_id": self.episode_id,
                "episode_reward": self.reward,
                "num_steps": len(self.step_logs),
                "steps": self.step_logs,
                "final_observation": self._obs,
            })


# ---------------------------------------------------------------------------
# Tokenization helpers (chat templating inside rollout_func)
# ---------------------------------------------------------------------------


def _tokenize_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    chat_template: str | None,
    chat_template_kwargs: dict,
    tools,
    add_generation_prompt: bool,
) -> list[int]:
    tokenized = tokenizer.apply_chat_template(
        conversation=[messages],
        tools=tools,
        chat_template=chat_template,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        padding=True,
        **chat_template_kwargs,
    )
    row_ids = tokenized["input_ids"][0]
    row_mask = tokenized["attention_mask"][0]
    return [int(t) for t, m in zip(row_ids, row_mask, strict=True) if m]


def _squeeze_vllm_logprobs(logprobs) -> list[float] | None:
    if logprobs is None:
        return None
    out: list[float] = []
    for seq in logprobs:
        for lp in seq:
            out.append(float(lp[0]) if lp and lp[0] is not None else 0.0)
    return out


def _step_max_new_tokens(obs: dict, trainer: "GRPOTrainer") -> int:
    cfg = RUNTIME_CFG
    global_max = int(trainer.args.max_completion_length)
    if cfg is None:
        return max(1, global_max)
    mode = resolve_budget_mode_from_observation(
        obs,
        default_mode=cfg.normalized_default_mode(),
        strict=cfg.strict_budget_mode_metadata,
    )
    m = int(cfg.max_tokens_per_step)
    if mode == "hard":
        rb = int(obs.get("remaining_budget", 0))
        m = min(m, max(0, rb))
    return max(1, min(m, global_max))


@contextmanager
def _temporary_vllm_max_tokens(trainer: "GRPOTrainer", max_tokens: int):
    vg = trainer.vllm_generation
    prev = vg.max_completion_length
    vg.max_completion_length = max_tokens
    try:
        yield
    finally:
        vg.max_completion_length = prev


def _pad_vllm_server_generates_to_target(
    trainer: "GRPOTrainer",
    local_generates: int,
    target: int,
    dummy_prompt_ids: list[int],
) -> None:
    """Pad with cheap dummy generates so every rank hits the same vLLM server collective count."""
    vg = trainer.vllm_generation
    if vg.mode != "server" or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for _ in range(max(0, target - local_generates)):
        with _temporary_vllm_max_tokens(trainer, 1):
            trainer.vllm_generation.generate(
                prompts=[dummy_prompt_ids],
                images=None,
                num_generations=1,
            )


def _rollout_one_episode(
    seed_messages: list,
    trainer: "GRPOTrainer",
    *,
    tok,
    chat_template,
    chat_template_kwargs: dict,
    tools,
    max_episode_turns: int,
    env_tokenizer_name: str,
    env_total_budget: int | None = None,
    model_profile: ResolvedProfile | None = None,
    per_episode_generate_cap: int | None = None,
) -> tuple[list[int], list[int], list[float], list[int], float, int]:
    messages = copy.deepcopy(seed_messages)
    with EpisodeSession(
        ENV_BASE_URL,
        tokenizer_name=env_tokenizer_name,
        total_budget=env_total_budget,
    ) as session:
        obs = session.reset_episode()
        if not isinstance(messages[-1].get("content"), str):
            raise TypeError(
                "rollout_func expects last message content to be a string for observation append."
            )
        messages[-1]["content"] = messages[-1]["content"] + format_observation_prompt(obs)

        prompt_ids_fixed = _tokenize_messages(
            tok,
            messages,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
            tools=tools,
            add_generation_prompt=True,
        )

        completion_ids: list[int] = []
        env_mask: list[int] = []
        logprob_seq: list[float] = []
        turns = 0
        effective_turn_limit = (
            min(max_episode_turns, per_episode_generate_cap)
            if per_episode_generate_cap is not None
            else max_episode_turns
        )

        while not session.done and turns < effective_turn_limit:
            turns += 1
            obs = session._obs or {}
            step_cap = _step_max_new_tokens(obs, trainer)
            before_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )

            with _temporary_vllm_max_tokens(trainer, step_cap):
                _, gen_ids_batch, logprobs_raw, _ = trainer.vllm_generation.generate(
                    prompts=[before_ids],
                    images=None,
                    num_generations=1,
                )
            gen_ids = gen_ids_batch[0]
            gen_lp = _squeeze_vllm_logprobs(logprobs_raw)
            if gen_lp is None or len(gen_lp) != len(gen_ids):
                gen_lp = [0.0] * len(gen_ids)

            completion_ids.extend(gen_ids)
            env_mask.extend([1] * len(gen_ids))
            logprob_seq.extend(gen_lp)

            text = tok.decode(gen_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            parsed = _parse_completion_for_profile(text, model_profile)
            step_md = _build_env_step_metadata(env_tokenizer_name, model_profile, parsed)
            log_extras: dict[str, Any] | None = None
            if model_profile and model_profile.output_parser:
                log_extras = {
                    "reasoning_trace": _truncate_for_log(parsed.reasoning) if parsed.reasoning else "",
                    "visible_response": _truncate_for_log(parsed.visible),
                }
            session.apply_response(text, step_metadata=step_md, log_extras=log_extras)

            if session.done:
                break

            after_asst_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            messages.append(
                {"role": "user", "content": format_observation_prompt(session._obs or {})}
            )
            after_user_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            suffix = after_user_ids[len(after_asst_ids) :]
            completion_ids.extend(suffix)
            env_mask.extend([0] * len(suffix))
            logprob_seq.extend([0.0] * len(suffix))

        return prompt_ids_fixed, completion_ids, logprob_seq, env_mask, float(session.reward), turns


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def build_rollout_func(
    max_episode_turns: int = 256,
    env_tokenizer_name: str | None = None,
    env_total_budget: int | None = None,
    fallback_model_id: str | None = None,
    model_profile: ResolvedProfile | None = None,
):
    """Build TRL ``rollout_func`` that steps OpenEnv explicitly (no tool ``solve`` parsing).

    When ``env_total_budget`` is ``None`` (default), the remote env computes a
    tokenizer-native budget from the sampled questions (if ``tokenizer_name`` is
    provided on reset).  Pass an explicit integer to override the env's budget
    computation entirely.
    """

    def rollout_func(prompts: list, trainer: "GRPOTrainer") -> dict[str, Any]:
        tok = trainer.processing_class
        chat_template = getattr(trainer, "chat_template", None)
        chat_kwargs = getattr(trainer, "chat_template_kwargs", None) or {}
        tools = getattr(trainer, "tools", None) or None
        env_tok = resolve_env_tokenizer_name(
            tok,
            trainer,
            env_tokenizer_name,
            fallback_model_id=fallback_model_id,
        )

        vg = trainer.vllm_generation
        dist_server_pad = (
            vg.mode == "server"
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        gen_cap = DIST_SERVER_GENERATES_PER_EPISODE if dist_server_pad else None
        dummy_prompt_ids: list[int] | None = None
        if dist_server_pad:
            dummy_prompt_ids = _tokenize_messages(
                tok,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "."},
                ],
                chat_template=chat_template,
                chat_template_kwargs=chat_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_env_mask: list[list[int]] = []
        all_env_reward: list[float] = []

        for seed_messages in prompts:
            p, c, lp, m, r, n_gen = _rollout_one_episode(
                seed_messages,
                trainer,
                tok=tok,
                chat_template=chat_template,
                chat_template_kwargs=chat_kwargs,
                tools=tools,
                max_episode_turns=max_episode_turns,
                env_tokenizer_name=env_tok,
                env_total_budget=env_total_budget,
                model_profile=model_profile,
                per_episode_generate_cap=gen_cap,
            )
            if dist_server_pad and dummy_prompt_ids is not None:
                _pad_vllm_server_generates_to_target(
                    trainer,
                    n_gen,
                    DIST_SERVER_GENERATES_PER_EPISODE,
                    dummy_prompt_ids,
                )
            all_prompt_ids.append(p)
            all_completion_ids.append(c)
            all_logprobs.append(lp)
            all_env_mask.append(m)
            all_env_reward.append(r)

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_mask": all_env_mask,
            "env_reward": all_env_reward,
        }

    return rollout_func


# ---------------------------------------------------------------------------
# Reward function for GRPOTrainer (rollout_func + extra_fields)
# ---------------------------------------------------------------------------


def reward_from_env(prompts, completions, completion_ids, **kwargs):
    """Return scalar episode reward from rollout ``env_reward`` extra field."""
    env_reward = kwargs.get("env_reward")
    if env_reward is not None:
        return [float(r) for r in env_reward]
    return [0.0] * len(prompts)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    global ENV_BASE_URL, RUNTIME_CFG, REWARD_LOG_PATH

    parser = argparse.ArgumentParser(description="GRPO training against remote OpenEnv env")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=4096)
    parser.add_argument("--max_tokens_per_step", type=int, default=2048)
    parser.add_argument("--min_tokens_per_step", type=int, default=10)
    parser.add_argument("--default_budget_mode", type=str, default="hard")
    parser.add_argument(
        "--strict_budget_mode_metadata",
        action="store_true",
        help="Require observation metadata budget_mode when resolving per-step caps.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--no_log_rewards", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--reward_log_path", type=str, default="")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"])
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM (server mode); default 1.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction for vLLM (server mode).",
    )
    parser.add_argument(
        "--vllm_server_host",
        type=str,
        default="127.0.0.1",
        help="Host of the trl vllm-serve process (server mode only). Default: 127.0.0.1",
    )
    parser.add_argument(
        "--vllm_server_port",
        type=int,
        default=8001,
        help=(
            "Port of the trl vllm-serve process (server mode only). "
            "Avoid 8000 if OpenEnv uses that port. Default: 8001."
        ),
    )
    parser.add_argument(
        "--vllm_group_port",
        type=int,
        default=51216,
        help="Port for TRL NCCL weight-sync group between training and vLLM. Default: 51216.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce training VRAM.",
    )
    parser.add_argument(
        "--no_bf16",
        action="store_true",
        help="Disable bfloat16 training (bf16 is on by default).",
    )
    parser.add_argument("--output_dir", type=str, default="runs/grpo_train")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--env_base_url", type=str, default=None)
    parser.add_argument("--space_url", type=str, default=None)
    parser.add_argument("--max_episode_turns", type=int, default=256)
    parser.add_argument(
        "--env_tokenizer_name",
        type=str,
        default=None,
        help=(
            "Hugging Face model id for AutoTokenizer on the remote env (aligns token counts with "
            "the policy). Required when training from a local checkpoint path; otherwise inferred "
            "from the tokenizer's name_or_path or --model."
        ),
    )
    parser.add_argument(
        "--env_total_budget",
        type=int,
        default=None,
        help=(
            "Explicit total token budget to send to the remote env on reset, overriding the env's "
            "own budget computation. When omitted, the env computes a tokenizer-native budget from "
            "the sampled questions (if tokenizer_name is provided) or falls back to config defaults."
        ),
    )
    parser.add_argument(
        "--model_profiles_path",
        type=str,
        default=None,
        help="Path to model_profiles.json (default: training/model_profiles.json next to this package).",
    )
    parser.add_argument(
        "--reasoning_mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Chat-template enable_thinking: auto uses profile JSON; on/off forces True/False.",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help=(
            "Load policy via Unsloth FastLanguageModel + LoRA (VRAM-friendly). "
            "With REPT_MODEL_SHARDING=1, use accelerate launch + FSDP yaml (experimental)."
        ),
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="With --use_unsloth: load base weights in 4-bit (QLoRA-style).",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank when --use_unsloth (default 16).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha when --use_unsloth (default 16).",
    )
    args = parser.parse_args()

    warn_unsloth_fsdp_experimental(use_unsloth=args.use_unsloth, load_in_4bit=args.load_in_4bit)

    if args.per_device_train_batch_size % args.num_generations != 0:
        raise SystemExit(
            "per_device_train_batch_size must be divisible by num_generations "
            f"({args.per_device_train_batch_size} % {args.num_generations} != 0)."
        )

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
        max_tokens_per_step=args.max_tokens_per_step,
        min_tokens_per_step=args.min_tokens_per_step,
        default_budget_mode=args.default_budget_mode,
        strict_budget_mode_metadata=args.strict_budget_mode_metadata,
    )
    REWARD_LOG_PATH = RUNTIME_CFG.resolved_reward_log_path(args.output_dir)
    if RUNTIME_CFG.log_rewards:
        print(f"Reward episode logs: {REWARD_LOG_PATH} (every {RUNTIME_CFG.log_every_n_steps} episodes)")

    profiles_path = Path(args.model_profiles_path) if args.model_profiles_path else None
    profile_registry = load_profiles(profiles_path)
    profile_lookup_id = profile_lookup_model_id(
        model_arg=args.model,
        env_tokenizer_name=args.env_tokenizer_name,
    )
    resolved_profile = profile_registry.resolve(profile_lookup_id)
    merged_chat_template_kwargs = merge_chat_template_kwargs_for_reasoning_mode(
        resolved_profile.chat_template_kwargs,
        reasoning_mode=args.reasoning_mode,
    )
    print(
        f"Model profile lookup={profile_lookup_id!r} → "
        f"parser={resolved_profile.output_parser!r}, "
        f"grading_use_visible_only={resolved_profile.grading_use_visible_only}, "
        f"chat_template_kwargs={merged_chat_template_kwargs!r}"
    )

    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Solve the next math problem under budget constraints."},
            ]
        ]
        * 100
    })

    policy_model: str | Any = args.model
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel, PatchFastRL
        except ImportError as e:
            raise SystemExit(
                "Unsloth is not installed. Install with: pip install unsloth"
            ) from e
        PatchFastRL("grpo")
        _fsdp = is_fsdp_model_sharding_env()
        _from_kw: dict[str, Any] = {
            "model_name": args.model,
            "max_seq_length": args.max_completion_length,
            "dtype": "auto",
            "load_in_4bit": args.load_in_4bit,
        }
        # Align with Accelerate FSDP CPU-efficient load path when sharding (all ranks construct).
        if _fsdp:
            _from_kw["low_cpu_mem_usage"] = True
        try:
            model, _tokenizer = FastLanguageModel.from_pretrained(**_from_kw)
        except TypeError:
            _from_kw.pop("low_cpu_mem_usage", None)
            model, _tokenizer = FastLanguageModel.from_pretrained(**_from_kw)
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=args.gradient_checkpointing,
            random_state=3407,
        )
        # PEFT + FSDP: inputs may need grads for checkpoint/wrapped forward (harmless if unused).
        if _fsdp and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        policy_model = model

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_group_port=args.vllm_group_port,
        num_train_epochs=args.num_train_epochs,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=not args.no_bf16,
        logging_steps=1,
        save_strategy="epoch",
        chat_template_kwargs=merged_chat_template_kwargs,
    )

    trainer = GRPOTrainer(
        model=policy_model,
        reward_funcs=reward_from_env,
        train_dataset=dataset,
        rollout_func=build_rollout_func(
            max_episode_turns=args.max_episode_turns,
            env_tokenizer_name=args.env_tokenizer_name,
            env_total_budget=args.env_total_budget,
            fallback_model_id=args.model,
            model_profile=resolved_profile,
        ),
        args=grpo_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
