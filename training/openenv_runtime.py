"""OpenEnv runtime helpers for loading env client symbols and base URLs."""

from __future__ import annotations

from importlib import import_module
from urllib.parse import urlparse


def to_openenv_base_url(
    *,
    env_base_url: str | None,
    space_url: str | None,
) -> str:
    """Resolve OpenEnv base URL from direct base URL or HF Space URL."""
    if env_base_url:
        return _normalize_base_url(env_base_url)
    if not space_url:
        raise ValueError("Set --env_base_url or --space_url.")
    return _space_url_to_base_url(space_url)


def _normalize_base_url(url: str) -> str:
    normalized = url.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError(f"Invalid env base URL: {url}")
    return normalized


def _space_url_to_base_url(space_url: str) -> str:
    """Convert HF Space URL to hf.space base URL."""
    s = space_url.strip().rstrip("/")
    parsed = urlparse(s if "://" in s else f"https://{s}")

    if parsed.netloc.endswith(".hf.space"):
        return _normalize_base_url(parsed.geturl())

    if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(
            "Unsupported --space_url format. Use a huggingface.co/spaces URL "
            "or an existing *.hf.space URL."
        )

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "spaces":
        raise ValueError(
            "Invalid space URL path. Expected: https://huggingface.co/spaces/<owner>/<space>"
        )

    owner, space = parts[1], parts[2]
    return f"https://{owner}-{space}.hf.space"


def load_reason_budget_symbols():
    """Load ReasonBudget env client + models from published env package."""
    # Preferred: explicit package module path if env repo later exposes one.
    client_module_candidates = [
        "reasoning_economics_env.client",
        "client",
    ]
    model_module_candidates = [
        "reasoning_economics_env.models",
        "models",
        "env.models",
    ]

    model_mod = _import_first_available(model_module_candidates)
    state_cls = getattr(model_mod, "ReasonBudgetState", None)

    try:
        action_cls = getattr(model_mod, "ReasonBudgetAction")
        obs_cls = getattr(model_mod, "ReasonBudgetObservation")
    except AttributeError as exc:
        raise ImportError(
            "Installed env package does not expose required ReasonBudget models. "
            "Install from HF Space artifact for prod or -e ../ReasoningEconomicsEnv for dev."
        ) from exc

    client_cls = None
    try:
        client_mod = _import_first_available(client_module_candidates)
        client_cls = getattr(client_mod, "ReasonBudgetEnvClient")
    except Exception:
        # Compatibility fallback when package exports models but not typed client.
        client_cls = _build_reason_budget_client_adapter(
            obs_cls=obs_cls,
            state_cls=state_cls,
        )

    return client_cls, action_cls, obs_cls


def resolve_budget_mode_from_observation(
    observation,
    *,
    default_mode: str = "hard",
    strict: bool = False,
) -> str:
    """Resolve budget mode from observation metadata with safe fallback."""
    mode = _normalize_mode(default_mode)
    metadata = getattr(observation, "metadata", None)
    if isinstance(metadata, dict):
        candidate = metadata.get("budget_mode")
        normalized = _normalize_mode(candidate)
        if normalized in {"hard", "soft"}:
            return normalized
    if strict:
        raise ValueError(
            "Observation metadata missing valid budget_mode. "
            "Expected metadata['budget_mode'] in {'hard','soft'}."
        )
    return mode


def _import_first_available(module_names: list[str]):
    errors: list[str] = []
    for mod in module_names:
        try:
            return import_module(mod)
        except Exception as exc:  # pragma: no cover - diagnostic only
            errors.append(f"{mod}: {exc}")
    joined = "; ".join(errors)
    raise ImportError(f"Unable to import expected env modules. Tried: {joined}")


def _normalize_mode(value) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"hard", "soft"}:
        return mode
    return "hard"


def _build_reason_budget_client_adapter(*, obs_cls, state_cls):
    """Build a minimal typed OpenEnv client if env artifact lacks one."""
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    class ReasonBudgetEnvClientAdapter(EnvClient):
        def _step_payload(self, action):
            return {"response": action.response}

        def _parse_result(self, payload):
            obs_data = payload.get("observation")
            if not isinstance(obs_data, dict):
                obs_data = payload if isinstance(payload, dict) else {}
            metadata = obs_data.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            done = payload.get("done", obs_data.get("done", False))
            reward = payload.get("reward", obs_data.get("reward"))
            observation = obs_cls(
                remaining_budget=obs_data.get("remaining_budget", 0.0),
                questions_remaining=obs_data.get("questions_remaining", 0),
                step_idx=obs_data.get("step_idx", 0),
                budget_per_remaining=obs_data.get("budget_per_remaining", 0.0),
                accuracy_so_far=obs_data.get("accuracy_so_far", 0.0),
                question=obs_data.get("question", ""),
                history=obs_data.get("history", []),
                done=done,
                reward=reward,
                metadata=metadata,
            )
            return StepResult(
                observation=observation,
                reward=reward,
                done=done,
            )

        def _parse_state(self, payload):
            state_data = payload.get("state")
            if not isinstance(state_data, dict):
                state_data = payload if isinstance(payload, dict) else {}
            if state_cls is None:
                return state_data
            return state_cls(
                episode_id=state_data.get("episode_id"),
                step_count=state_data.get("step_count", 0),
                total_budget=state_data.get("total_budget", 0),
                spent_budget=state_data.get("spent_budget", 0),
                questions_answered=state_data.get("questions_answered", 0),
                total_correct=state_data.get("total_correct", 0),
                current_accuracy=state_data.get("current_accuracy", 0.0),
                budget_remaining_ratio=state_data.get("budget_remaining_ratio", 0.0),
            )

    ReasonBudgetEnvClientAdapter.__name__ = "ReasonBudgetEnvClientAdapter"
    return ReasonBudgetEnvClientAdapter
