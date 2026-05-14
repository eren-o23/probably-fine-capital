"""Shared LLM call helper for all analyst agents.

Wraps the Fireworks AI API (OpenAI-compatible) with JSON parsing,
Pydantic validation, one retry on parse failure, and safe-None return.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """Return the module-level AsyncOpenAI client, creating it on first call."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=config.FIREWORKS_API_KEY,
            base_url=config.FIREWORKS_BASE_URL,
        )
    return _client


async def call_llm(
    prompt: str,
    response_model: type[T],
    system_prompt: str = (
        "You are a financial analyst. "
        "Respond in valid JSON only. No markdown, no explanation outside the JSON."
    ),
) -> T | None:
    """Call the Fireworks LLM and parse the response as a Pydantic model.

    Attempts the call once, retries once on JSON/validation failure.
    Returns None if both attempts fail — callers must handle None.

    Args:
        prompt: The user message content.
        response_model: Pydantic model class to parse the response into.
        system_prompt: System instruction. Defaults to JSON-only financial analyst.

    Returns:
        A validated instance of response_model, or None on failure.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(2):
        try:
            response = await client.chat.completions.create(
                model=config.FIREWORKS_MODEL,
                messages=messages,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            return response_model.model_validate(parsed)
        except json.JSONDecodeError as exc:
            logger.warning(
                "call_llm: JSON parse error on attempt %d for %s: %s",
                attempt + 1,
                response_model.__name__,
                exc,
            )
        except ValidationError as exc:
            logger.warning(
                "call_llm: validation error on attempt %d for %s: %s",
                attempt + 1,
                response_model.__name__,
                exc,
            )
        except Exception as exc:
            logger.error(
                "call_llm: unexpected error on attempt %d for %s: %s",
                attempt + 1,
                response_model.__name__,
                exc,
            )
            return None  # network/auth errors — no point retrying

    logger.error(
        "call_llm: both attempts failed for model %s", response_model.__name__
    )
    return None
