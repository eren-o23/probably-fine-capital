"""Shared LLM call helper for all analyst agents.

Wraps the Fireworks AI API (OpenAI-compatible) with JSON parsing,
Pydantic validation, one retry on parse failure, and safe-None return.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TypeVar

from openai import AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_client: AsyncOpenAI | None = None
_LLM_SEMAPHORE = asyncio.Semaphore(1)
_MAX_ATTEMPTS = 3


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

    At most 5 concurrent HTTP calls are allowed via _LLM_SEMAPHORE.
    On 429 rate-limit errors, retries with exponential backoff (2s, 4s).
    Returns None if all _MAX_ATTEMPTS fail or a non-retryable error occurs.

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

    for attempt in range(_MAX_ATTEMPTS):
        try:
            async with _LLM_SEMAPHORE:
                response = await client.chat.completions.create(
                    model=config.FIREWORKS_MODEL,
                    messages=messages,
                    max_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE,
                )
                await asyncio.sleep(7.0)
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                content = getattr(choice.message, "reasoning_content", None)
            if not content:
                logger.error(
                    "call_llm: no content in response for %s",
                    response_model.__name__,
                )
                return None
            original_content = content
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if not content:
                think_match = re.search(r"<think>(.*?)</think>", original_content, re.DOTALL)
                if think_match:
                    content = think_match.group(1).strip()
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
            else:
                logger.error(
                    "call_llm: no JSON object found in response for %s",
                    response_model.__name__,
                )
                return None
            parsed = json.loads(content)
            return response_model.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "call_llm: JSON/validation error on attempt %d for %s: %s",
                attempt + 1,
                response_model.__name__,
                exc,
            )
        except RateLimitError:
            if attempt < _MAX_ATTEMPTS - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "call_llm: rate limited, retrying in %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "call_llm: rate limited on all %d attempts for %s",
                    _MAX_ATTEMPTS,
                    response_model.__name__,
                )
                return None
        except Exception as exc:
            logger.error(
                "call_llm: unexpected error on attempt %d for %s: %s",
                attempt + 1,
                response_model.__name__,
                exc,
            )
            return None

    logger.error(
        "call_llm: all %d attempts failed for %s", _MAX_ATTEMPTS, response_model.__name__
    )
    return None
