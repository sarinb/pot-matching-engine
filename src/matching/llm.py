"""Centralized helpers for Anthropic LLM calls."""

from __future__ import annotations

import json
import logging
import re

from anthropic import Anthropic

from src.matching.config import settings

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def call_llm_json(
    client: Anthropic,
    system: str,
    user: str,
    *,
    fast: bool = True,
) -> dict:
    model = settings.anthropic_fast_model if fast else settings.anthropic_model
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON (%s model): %s", model, raw[:200])
        return {}


def call_llm_text(
    client: Anthropic,
    system: str,
    user: str,
    *,
    fast: bool = False,
) -> str:
    model = settings.anthropic_fast_model if fast else settings.anthropic_model
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()
