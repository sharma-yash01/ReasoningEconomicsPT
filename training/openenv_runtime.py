"""OpenEnv runtime helpers: generic dict-based client and URL resolution."""

from __future__ import annotations

from urllib.parse import urlparse

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient


class ReasonBudgetClient(EnvClient):
    """Dict-in / dict-out OpenEnv client with no domain-specific Pydantic models."""

    def _step_payload(self, action: dict) -> dict:
        return action

    def _parse_result(self, payload: dict) -> StepResult:
        obs = payload.get("observation", payload)
        if not isinstance(obs, dict):
            obs = payload
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.get("reward")),
            done=payload.get("done", obs.get("done", False)),
        )

    def _parse_state(self, payload: dict) -> dict:
        return payload.get("state", payload)


def to_openenv_base_url(
    *,
    env_base_url: str | None,
    space_url: str | None,
):
    """Resolve OpenEnv base URL from direct base URL or HF Space URL."""
    if env_base_url:
        return _normalize_base_url(env_base_url)
    if not space_url:
        raise ValueError("Set --env_base_url or --space_url.")
    return _space_url_to_base_url(space_url)


def _normalize_base_url(url: str):
    normalized = url.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError(f"Invalid env base URL: {url}")
    return normalized


def _space_url_to_base_url(space_url: str):
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


def resolve_budget_mode_from_observation(
    observation: dict,
    *,
    default_mode: str = "hard",
    strict: bool = False,
):
    """Resolve budget mode from observation metadata with safe fallback."""
    mode = _normalize_mode(default_mode)
    metadata = observation.get("metadata") if isinstance(observation, dict) else None
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


def _normalize_mode(value):
    mode = str(value or "").strip().lower()
    if mode in {"hard", "soft"}:
        return mode
    return "hard"
