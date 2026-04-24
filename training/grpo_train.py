"""GRPO training with TRL rollout_func and remote OpenEnv (explicit env stepping)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from training.config import TrainingRuntimeConfig
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
ROLLOUT_DEBUG: bool = False
ROLLOUT_DEBUG_PATH: str = ""
ROLLOUT_DEBUG_COUNT: int = 0
# If OpenEnv rejects a step, keep the run moving and give that episode a penalty.
ENV_STEP_ERROR_PENALTY: float = -0.4

LOG_PREVIEW_CHARS = 2000


def _truncate_for_log(s: str, max_len: int = LOG_PREVIEW_CHARS) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…[truncated]"


def _debug_rollout(event: str, **fields: Any) -> None:
    """Write optional JSONL events while an episode is running.

    The reward log only shows finished episodes. This file shows reset,
    generation, env step, and error events when a run hangs or crashes.
    """
    global ROLLOUT_DEBUG_COUNT
    if not ROLLOUT_DEBUG:
        return

    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with LOG_LOCK:
        ROLLOUT_DEBUG_COUNT += 1
        entry["debug_index"] = ROLLOUT_DEBUG_COUNT
        line = json.dumps(entry, ensure_ascii=True, sort_keys=True)
        print(f"[rollout-debug] {line}", flush=True)
        if ROLLOUT_DEBUG_PATH:
            path = Path(ROLLOUT_DEBUG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


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
    """Append one episode-level reward record to reward_log.jsonl."""
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
            "questions_remaining_after": self._obs.get("questions_remaining") if self._obs else None,
            "remaining_budget_after": self._obs.get("remaining_budget") if self._obs else None,
            "done_after_step": self.done,
        }
        if log_extras:
            entry.update(log_extras)
        self.step_logs.append(entry)


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
) -> tuple[list[int], list[int], list[float], list[int], float]:
    messages = copy.deepcopy(seed_messages)
    episode_debug_id = uuid4().hex[:12]
    _debug_rollout(
        "episode_enter",
        episode_debug_id=episode_debug_id,
        max_episode_turns=max_episode_turns,
        env_tokenizer_name=env_tokenizer_name,
        env_total_budget=env_total_budget,
        trainer_max_completion_length=int(trainer.args.max_completion_length),
        prompt_message_count=len(messages),
    )
    with EpisodeSession(
        ENV_BASE_URL,
        tokenizer_name=env_tokenizer_name,
        total_budget=env_total_budget,
    ) as session:
        reset_t0 = time.monotonic()
        obs = session.reset_episode()
        _debug_rollout(
            "episode_reset_done",
            episode_debug_id=episode_debug_id,
            episode_id=session.episode_id,
            elapsed_s=round(time.monotonic() - reset_t0, 3),
            questions_remaining=obs.get("questions_remaining"),
            remaining_budget=obs.get("remaining_budget"),
            budget_per_remaining=obs.get("budget_per_remaining"),
            question_chars=len(str(obs.get("question", ""))),
        )
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
        initial_questions_remaining = int(obs.get("questions_remaining", 0))
        _debug_rollout(
            "episode_prompt_ready",
            episode_debug_id=episode_debug_id,
            episode_id=session.episode_id,
            prompt_tokens=len(prompt_ids_fixed),
            initial_questions_remaining=initial_questions_remaining,
        )

        completion_ids: list[int] = []
        env_mask: list[int] = []
        logprob_seq: list[float] = []
        turns = 0
        any_step_hit_generation_cap = False
        termination_reason = "max_episode_turns"

        while not session.done and turns < max_episode_turns:
            turns += 1
            obs = session._obs or {}
            step_cap = _step_max_new_tokens(obs, trainer)
            _debug_rollout(
                "turn_start",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                questions_remaining=obs.get("questions_remaining"),
                remaining_budget=obs.get("remaining_budget"),
                tokens_used=obs.get("tokens_used"),
                step_cap=step_cap,
                completion_tokens_so_far=int(sum(env_mask)),
                serialized_tokens_so_far=len(completion_ids),
                message_count=len(messages),
            )
            before_ids = _tokenize_messages(
                tok,
                messages,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
                tools=tools,
                add_generation_prompt=True,
            )
            _debug_rollout(
                "turn_prompt_tokenized",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                before_prompt_tokens=len(before_ids),
            )

            gen_t0 = time.monotonic()
            try:
                _debug_rollout(
                    "turn_generate_start",
                    episode_debug_id=episode_debug_id,
                    episode_id=session.episode_id,
                    turn=turns,
                    step_cap=step_cap,
                    before_prompt_tokens=len(before_ids),
                )
                with _temporary_vllm_max_tokens(trainer, step_cap):
                    _, gen_ids_batch, logprobs_raw, _ = trainer.vllm_generation.generate(
                        prompts=[before_ids],
                        images=None,
                        num_generations=1,
                    )
            except Exception as exc:
                _debug_rollout(
                    "turn_generate_error",
                    episode_debug_id=episode_debug_id,
                    episode_id=session.episode_id,
                    turn=turns,
                    elapsed_s=round(time.monotonic() - gen_t0, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                raise
            gen_ids = gen_ids_batch[0]
            gen_lp = _squeeze_vllm_logprobs(logprobs_raw)
            if gen_lp is None or len(gen_lp) != len(gen_ids):
                gen_lp = [0.0] * len(gen_ids)
            step_hit_generation_cap = len(gen_ids) == step_cap
            any_step_hit_generation_cap = any_step_hit_generation_cap or step_hit_generation_cap
            _debug_rollout(
                "turn_generate_done",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                elapsed_s=round(time.monotonic() - gen_t0, 3),
                generated_tokens=len(gen_ids),
                step_cap=step_cap,
                hit_generation_cap=step_hit_generation_cap,
            )

            completion_ids.extend(gen_ids)
            env_mask.extend([1] * len(gen_ids))
            logprob_seq.extend(gen_lp)

            text = tok.decode(gen_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": text})
            parsed = _parse_completion_for_profile(text, model_profile)
            _debug_rollout(
                "turn_decoded",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                response_chars=len(text),
                visible_chars=len(parsed.visible),
                reasoning_chars=len(parsed.reasoning),
                visible_preview=_truncate_for_log(parsed.visible, 200),
            )
            step_md = _build_env_step_metadata(env_tokenizer_name, model_profile, parsed)
            log_extras: dict[str, Any] | None = None
            if model_profile and model_profile.output_parser:
                log_extras = {
                    "reasoning_trace": _truncate_for_log(parsed.reasoning) if parsed.reasoning else "",
                    "visible_response": _truncate_for_log(parsed.visible),
                }
            if log_extras is None:
                log_extras = {}
            log_extras.update(
                {
                    "step_completion_tokens": len(gen_ids),
                    "step_max_tokens": step_cap,
                    "step_hit_generation_cap": step_hit_generation_cap,
                    "completion_tokens_so_far": int(sum(env_mask)),
                    "remaining_rollout_after_step": int(trainer.args.max_completion_length) - len(completion_ids),
                }
            )
            step_t0 = time.monotonic()
            try:
                _debug_rollout(
                    "turn_env_step_start",
                    episode_debug_id=episode_debug_id,
                    episode_id=session.episode_id,
                    turn=turns,
                    response_chars=len(text),
                    metadata_keys=sorted(step_md.keys()),
                )
                session.apply_response(text, step_metadata=step_md, log_extras=log_extras)
            except Exception as exc:
                # The model already generated text, so turn the env failure into
                # a logged penalty instead of dropping the whole distributed run.
                penalty = float(ENV_STEP_ERROR_PENALTY)
                _debug_rollout(
                    "turn_env_step_error",
                    episode_debug_id=episode_debug_id,
                    episode_id=session.episode_id,
                    turn=turns,
                    elapsed_s=round(time.monotonic() - step_t0, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    penalty=penalty,
                    traceback=traceback.format_exc(),
                )
                session.reward += penalty
                session.done = True
                termination_reason = "env_step_error"
                error_entry: dict[str, Any] = {
                    "step_index": len(session.step_logs) + 1,
                    "question": (session._obs or {}).get("question", ""),
                    "question_summary": (session._obs or {}).get("question_summary", ""),
                    "questions_remaining_before": (session._obs or {}).get("questions_remaining"),
                    "remaining_budget_before": (session._obs or {}).get("remaining_budget"),
                    "tokens_used_before": (session._obs or {}).get("tokens_used"),
                    "model_response": text,
                    "raw_step_reward": penalty,
                    "scaled_step_reward": penalty,
                    "questions_remaining_after": (session._obs or {}).get("questions_remaining"),
                    "remaining_budget_after": (session._obs or {}).get("remaining_budget"),
                    "done_after_step": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "termination_reason": termination_reason,
                }
                if log_extras:
                    error_entry.update(log_extras)
                session.step_logs.append(error_entry)
                break
            _debug_rollout(
                "turn_env_step_done",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                elapsed_s=round(time.monotonic() - step_t0, 3),
                done=session.done,
                cumulative_reward=session.reward,
                questions_remaining=(session._obs or {}).get("questions_remaining"),
                remaining_budget=(session._obs or {}).get("remaining_budget"),
                step_logs=len(session.step_logs),
            )

            if session.done:
                termination_reason = "env_done"
                _debug_rollout(
                    "episode_env_done",
                    episode_debug_id=episode_debug_id,
                    episode_id=session.episode_id,
                    turn=turns,
                    cumulative_reward=session.reward,
                )
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
            _debug_rollout(
                "turn_user_suffix_appended",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turn=turns,
                after_assistant_tokens=len(after_asst_ids),
                after_user_tokens=len(after_user_ids),
                env_suffix_tokens=len(suffix),
                serialized_tokens_so_far=len(completion_ids),
            )

        if not session.done and turns >= max_episode_turns:
            _debug_rollout(
                "episode_max_turns",
                episode_debug_id=episode_debug_id,
                episode_id=session.episode_id,
                turns=turns,
                max_episode_turns=max_episode_turns,
                cumulative_reward=session.reward,
                questions_remaining=(session._obs or {}).get("questions_remaining"),
            )

        final_observation = session._obs or {}
        final_questions_remaining = int(final_observation.get("questions_remaining", 0))
        questions_completed = max(0, initial_questions_remaining - final_questions_remaining)
        _debug_rollout(
            "episode_log_write_start",
            episode_debug_id=episode_debug_id,
            episode_id=session.episode_id,
            termination_reason=termination_reason,
            questions_completed=questions_completed,
            final_questions_remaining=final_questions_remaining,
            total_completion_tokens=int(sum(env_mask)),
            total_tokens_serialized=len(completion_ids),
        )
        _write_episode_log(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "episode_end",
                "episode_id": session.episode_id,
                "episode_reward": session.reward,
                "episode_weighted_reward": session.reward,
                "num_steps": len(session.step_logs),
                "num_steps_executed": len(session.step_logs),
                "steps": session.step_logs,
                "final_observation": final_observation,
                "initial_questions_remaining": initial_questions_remaining,
                "final_questions_remaining": final_questions_remaining,
                "questions_completed": questions_completed,
                "total_completion_tokens": int(sum(env_mask)),
                "total_tokens_serialized": len(completion_ids),
                "prompt_tokens": len(prompt_ids_fixed),
                "any_step_hit_generation_cap": any_step_hit_generation_cap,
                "episode_clipped": termination_reason in {"max_episode_turns"},
                "termination_reason": termination_reason,
                "max_completion_length": int(trainer.args.max_completion_length),
            }
        )
        _debug_rollout(
            "episode_return",
            episode_debug_id=episode_debug_id,
            episode_id=session.episode_id,
            reward=session.reward,
            prompt_tokens=len(prompt_ids_fixed),
            completion_tokens=len(completion_ids),
            env_mask_tokens=int(sum(env_mask)),
            logprobs=len(logprob_seq),
        )

        return prompt_ids_fixed, completion_ids, logprob_seq, env_mask, float(session.reward)


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

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_env_mask: list[list[int]] = []
        all_env_reward: list[float] = []

        for seed_messages in prompts:
            p, c, lp, m, r = _rollout_one_episode(
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
    global ENV_BASE_URL, RUNTIME_CFG, REWARD_LOG_PATH, ROLLOUT_DEBUG, ROLLOUT_DEBUG_PATH
    global ENV_STEP_ERROR_PENALTY

    parser = argparse.ArgumentParser(description="GRPO training against remote OpenEnv env")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_prompts", type=int, default=100)
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
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max model length/context cap; forwarded when supported by TRL.",
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
        "--debug_rollout",
        action="store_true",
        help=(
            "Print and write JSONL rollout progress events around generation and env steps. "
            "Also enabled by REPT_DEBUG_ROLLOUT=1."
        ),
    )
    parser.add_argument(
        "--rollout_debug_path",
        type=str,
        default="",
        help=(
            "Optional JSONL path for rollout debug events. Defaults to "
            "<output_dir>/rollout_debug.jsonl when debug is enabled."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help=(
            "Total training steps passed to GRPOConfig. Required when FSDP is active: "
            "TRL wraps the dataset as IterableDataset (no __len__) and Transformers Trainer "
            "requires max_steps when the dataloader has no length."
        ),
    )
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,
        help="FSDP strategy string passed to GRPOConfig (e.g. 'full_shard auto_wrap').",
    )
    parser.add_argument(
        "--fsdp_config",
        type=str,
        default=None,
        help="JSON string dict for FSDP config passed to GRPOConfig.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed ZeRO config JSON passed to GRPOConfig.",
    )
    parser.add_argument(
        "--vllm_enable_sleep_mode",
        action="store_true",
        help="Enable vLLM sleep mode during optimizer step (colocate only; reduces memory pressure).",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="TrainingArguments save_strategy. Use 'no' for metric-only smoke or curve runs.",
    )
    parser.add_argument(
        "--skip_final_save",
        action="store_true",
        help="Skip trainer.save_model() at the end of training.",
    )
    parser.add_argument(
        "--env_step_error_penalty",
        type=float,
        default=-0.4,
        help=(
            "Episode reward assigned when OpenEnv step() raises, allowing distributed "
            "training to continue instead of crashing all ranks."
        ),
    )
    args = parser.parse_args()
    ENV_STEP_ERROR_PENALTY = float(args.env_step_error_penalty)

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
    ROLLOUT_DEBUG = args.debug_rollout or os.environ.get("REPT_DEBUG_ROLLOUT", "0") == "1"
    debug_path = args.rollout_debug_path or os.environ.get("REPT_ROLLOUT_DEBUG_PATH", "")
    ROLLOUT_DEBUG_PATH = debug_path or str(Path(args.output_dir) / "rollout_debug.jsonl")
    if ROLLOUT_DEBUG:
        print(f"Rollout debug logs: {ROLLOUT_DEBUG_PATH}", flush=True)

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

    if args.n_prompts < 1:
        raise SystemExit(f"--n_prompts must be >= 1 (got {args.n_prompts})")

    dataset = Dataset.from_dict({
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Solve the next math problem under budget constraints."},
            ]
        ]
        * args.n_prompts
    })

    grpo_kwargs = {
        "output_dir": args.output_dir,
        "use_vllm": True,
        "vllm_mode": args.vllm_mode,
        "vllm_tensor_parallel_size": args.vllm_tensor_parallel_size,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_server_host": args.vllm_server_host,
        "vllm_server_port": args.vllm_server_port,
        "vllm_group_port": args.vllm_group_port,
        "num_train_epochs": args.num_train_epochs,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": not args.no_bf16,
        "logging_steps": 1,
        "save_strategy": args.save_strategy,
        "chat_template_kwargs": merged_chat_template_kwargs,
    }
    if args.max_steps > 0:
        grpo_kwargs["max_steps"] = args.max_steps
    if args.vllm_max_model_len is not None:
        import inspect

        if "vllm_max_model_length" in inspect.signature(GRPOConfig).parameters:
            grpo_kwargs["vllm_max_model_length"] = args.vllm_max_model_len
        else:
            print(
                "[WARN] Installed TRL GRPOConfig does not expose vllm_max_model_length; "
                "ignoring --vllm_max_model_len.",
                flush=True,
            )
    if args.fsdp is not None or args.fsdp_config is not None or args.deepspeed is not None \
            or args.vllm_enable_sleep_mode:
        import inspect as _inspect
        import json as _json
        _sig = _inspect.signature(GRPOConfig).parameters

        if args.fsdp is not None:
            if "fsdp" not in _sig:
                raise SystemExit(
                    "[ERROR] --fsdp was requested but installed TRL GRPOConfig does not expose 'fsdp'. "
                    "Cannot proceed: sharding would silently fall back to plain DDP and reproduce the OOM. "
                    "Upgrade TRL or check the installed version."
                )
            grpo_kwargs["fsdp"] = args.fsdp
            # When FSDP is active, activation_checkpointing is passed via fsdp_config.
            # TrainingArguments gradient_checkpointing triggers a redundant AllGather in
            # the backward pass under FSDP; disable it here.
            # See: https://github.com/huggingface/transformers/issues/30404
            if grpo_kwargs.get("gradient_checkpointing"):
                grpo_kwargs["gradient_checkpointing"] = False
                print(
                    "[INFO] FSDP mode: disabling gradient_checkpointing in TrainingArguments; "
                    "activation_checkpointing is set via fsdp_config instead.",
                    flush=True,
                )
        if args.fsdp_config is not None:
            if "fsdp_config" not in _sig:
                raise SystemExit(
                    "[ERROR] --fsdp_config was requested but GRPOConfig does not expose 'fsdp_config'."
                )
            grpo_kwargs["fsdp_config"] = _json.loads(args.fsdp_config)
        if args.deepspeed is not None:
            if "deepspeed" not in _sig:
                raise SystemExit(
                    "[ERROR] --deepspeed was requested but installed TRL GRPOConfig does not expose 'deepspeed'. "
                    "Cannot proceed: sharding would silently fall back to plain DDP. "
                    "Upgrade TRL or check the installed version."
                )
            grpo_kwargs["deepspeed"] = args.deepspeed
        if args.vllm_enable_sleep_mode:
            if "vllm_enable_sleep_mode" not in _sig:
                raise SystemExit(
                    "[ERROR] --vllm_enable_sleep_mode was requested but installed TRL GRPOConfig "
                    "does not expose 'vllm_enable_sleep_mode'. Cannot proceed: colocated vLLM "
                    "would stay resident during optimizer step and may reproduce the OOM."
                )
            grpo_kwargs["vllm_enable_sleep_mode"] = True
    grpo_config = GRPOConfig(**grpo_kwargs)
    print(
        "Sharding config:",
        "fsdp=", getattr(grpo_config, "fsdp", None),
        "fsdp_config=", getattr(grpo_config, "fsdp_config", None),
        "deepspeed=", getattr(grpo_config, "deepspeed", None),
        "vllm_sleep=", getattr(grpo_config, "vllm_enable_sleep_mode", None),
        flush=True,
    )

    trainer = GRPOTrainer(
        model=args.model,
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
    if args.skip_final_save:
        print(f"Training complete. Final model save skipped. Artifacts at {args.output_dir}")
    else:
        trainer.save_model(args.output_dir)
        print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
