"""Per-model profiles: chat_template_kwargs, output parsing (Qwen think blocks), grading hints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Qwen3 / Qwen3.5 hybrid thinking delimiters (see HF model card / tokenizer chat template).
_DEFAULT_QWEN_THINK_OPEN = "<think>"
_DEFAULT_QWEN_THINK_CLOSE = "</think>"


@dataclass
class ParsedCompletion:
    """Decoded model completion split into reasoning vs visible tail."""

    full: str
    reasoning: str
    visible: str


@dataclass
class ResolvedProfile:
    """Merged profile for a specific model id lookup."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    output_parser: str | None = None
    think_tag_open: str | None = None
    think_tag_close: str | None = None
    grading_use_visible_only: bool = False


@dataclass
class _RawProfileRule:
    match_exact: list[str]
    match_prefixes: list[str]
    chat_template_kwargs: dict[str, Any]
    output_parser: str | None
    think_tag_open: str | None
    think_tag_close: str | None
    grading_use_visible_only: bool


class ModelProfileRegistry:
    def __init__(
        self,
        default: ResolvedProfile,
        rules: list[_RawProfileRule],
    ):
        self.default = default
        self._rules = rules

    def resolve(self, model_id: str) -> ResolvedProfile:
        mid = (model_id or "").strip()
        if not mid:
            return ResolvedProfile(
                chat_template_kwargs=dict(self.default.chat_template_kwargs),
                output_parser=self.default.output_parser,
                think_tag_open=self.default.think_tag_open,
                think_tag_close=self.default.think_tag_close,
                grading_use_visible_only=self.default.grading_use_visible_only,
            )

        for rule in self._rules:
            if mid in rule.match_exact:
                return self._merge_default(rule)

        best: _RawProfileRule | None = None
        best_len = -1
        for rule in self._rules:
            for pfx in rule.match_prefixes:
                if mid.startswith(pfx) and len(pfx) > best_len:
                    best = rule
                    best_len = len(pfx)

        if best is not None:
            return self._merge_default(best)

        return ResolvedProfile(
            chat_template_kwargs=dict(self.default.chat_template_kwargs),
            output_parser=self.default.output_parser,
            think_tag_open=self.default.think_tag_open,
            think_tag_close=self.default.think_tag_close,
            grading_use_visible_only=self.default.grading_use_visible_only,
        )

    def _merge_default(self, rule: _RawProfileRule) -> ResolvedProfile:
        kwargs = dict(self.default.chat_template_kwargs)
        kwargs.update(rule.chat_template_kwargs)
        return ResolvedProfile(
            chat_template_kwargs=kwargs,
            output_parser=rule.output_parser if rule.output_parser is not None else self.default.output_parser,
            think_tag_open=rule.think_tag_open if rule.think_tag_open is not None else self.default.think_tag_open,
            think_tag_close=rule.think_tag_close if rule.think_tag_close is not None else self.default.think_tag_close,
            grading_use_visible_only=rule.grading_use_visible_only,
        )


def load_profiles(path: Path | None = None) -> ModelProfileRegistry:
    if path is None:
        path = Path(__file__).resolve().parent / "model_profiles.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    default_block = raw.get("default") or {}
    default = ResolvedProfile(
        chat_template_kwargs=dict(default_block.get("chat_template_kwargs") or {}),
        output_parser=default_block.get("output_parser"),
        think_tag_open=default_block.get("think_tag_open"),
        think_tag_close=default_block.get("think_tag_close"),
        grading_use_visible_only=bool(default_block.get("grading_use_visible_only", False)),
    )
    rules: list[_RawProfileRule] = []
    for entry in raw.get("profiles") or []:
        rules.append(
            _RawProfileRule(
                match_exact=[str(x).strip() for x in (entry.get("match_exact") or []) if str(x).strip()],
                match_prefixes=[str(x).strip() for x in (entry.get("match_prefixes") or []) if str(x).strip()],
                chat_template_kwargs=dict(entry.get("chat_template_kwargs") or {}),
                output_parser=entry.get("output_parser"),
                think_tag_open=entry.get("think_tag_open"),
                think_tag_close=entry.get("think_tag_close"),
                grading_use_visible_only=bool(entry.get("grading_use_visible_only", False)),
            )
        )
    return ModelProfileRegistry(default=default, rules=rules)


def parse_completion(
    text: str,
    parser_id: str | None,
    *,
    think_tag_open: str | None = None,
    think_tag_close: str | None = None,
) -> ParsedCompletion:
    if not parser_id or parser_id == "none":
        t = text or ""
        return ParsedCompletion(full=t, reasoning="", visible=t)

    if parser_id == "qwen3_think":
        open_tag = think_tag_open or _DEFAULT_QWEN_THINK_OPEN
        close_tag = think_tag_close or _DEFAULT_QWEN_THINK_CLOSE
        return _parse_qwen_think(text or "", open_tag, close_tag)

    raise ValueError(f"Unknown output_parser: {parser_id!r}")


def _parse_qwen_think(text: str, open_tag: str, close_tag: str) -> ParsedCompletion:
    if open_tag not in text:
        return ParsedCompletion(full=text, reasoning="", visible=text)

    first = text.find(open_tag)
    after_open = first + len(open_tag)
    close_idx = text.find(close_tag, after_open)
    if close_idx == -1:
        # Unclosed think block: treat everything after open as reasoning, no visible tail.
        reasoning = text[after_open:].strip()
        return ParsedCompletion(full=text, reasoning=reasoning, visible="")

    reasoning = text[after_open:close_idx].strip()
    visible = text[close_idx + len(close_tag) :].lstrip()
    return ParsedCompletion(full=text, reasoning=reasoning, visible=visible or text)


def merge_chat_template_kwargs_for_reasoning_mode(
    base: dict[str, Any],
    *,
    reasoning_mode: str,
) -> dict[str, Any]:
    """Override enable_thinking when reasoning_mode is on or off; auto leaves base unchanged."""
    mode = (reasoning_mode or "auto").strip().lower()
    out = dict(base)
    if mode == "on":
        out["enable_thinking"] = True
    elif mode == "off":
        out["enable_thinking"] = False
    elif mode != "auto":
        raise ValueError(f"reasoning_mode must be auto, on, or off (got {reasoning_mode!r})")
    return out


def profile_lookup_model_id(
    *,
    model_arg: str,
    env_tokenizer_name: str | None,
) -> str:
    """Hub-style id for profile matching when --model is a local checkpoint path."""
    p = Path(model_arg)
    if p.is_absolute() or model_arg.startswith(("./", ".\\")):
        if env_tokenizer_name and str(env_tokenizer_name).strip():
            return str(env_tokenizer_name).strip()
        return model_arg
    return model_arg.strip()
