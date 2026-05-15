"""Shared LLM call helper for all analyst agents.

Wraps the Fireworks AI API (OpenAI-compatible) with JSON parsing,
Pydantic validation, one retry on parse failure, and safe-None return.
"""

from __future__ import annotations

import ast
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
            # Step 1: extract raw text
            choice = response.choices[0]
            raw = choice.message.content
            if not raw:
                raw = getattr(choice.message, "reasoning_content", None)
            if not raw:
                logger.error(
                    "call_llm: no content in response on attempt %d for %s",
                    attempt + 1,
                    response_model.__name__,
                )
                return None

            # Step 2: strip <think> blocks; if nothing remains, fall back to inside them
            text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if not text:
                think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
                if think_match:
                    text = think_match.group(1).strip()
                    logger.debug(
                        "call_llm: fell back to <think> block content for %s",
                        response_model.__name__,
                    )

            # Step 3: extract first {...} block
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if not json_match:
                logger.error(
                    "call_llm: no JSON object found on attempt %d for %s — text: %.200s",
                    attempt + 1,
                    response_model.__name__,
                    text,
                )
                return None
            extracted = json_match.group(0)

            # Step 4: clean (strip comments and trailing commas) then parse
            cleaned = re.sub(r"//.*$", "", extracted, flags=re.MULTILINE)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "call_llm: json.loads failed on attempt %d for %s (%s) — trying quote fix",
                    attempt + 1,
                    response_model.__name__,
                    exc,
                )
                quote_fixed = re.sub(r"(?<!\\)'", '"', cleaned)
                try:
                    parsed = json.loads(quote_fixed)
                except json.JSONDecodeError as exc2:
                    logger.warning(
                        "call_llm: quote fix failed on attempt %d for %s (%s) — trying ast.literal_eval",
                        attempt + 1,
                        response_model.__name__,
                        exc2,
                    )
                    try:
                        parsed = ast.literal_eval(extracted)
                    except (ValueError, SyntaxError) as exc3:
                        logger.warning(
                            "call_llm: ast.literal_eval failed on attempt %d for %s: %s",
                            attempt + 1,
                            response_model.__name__,
                            exc3,
                        )
                        raise json.JSONDecodeError(str(exc3), extracted, 0) from exc3

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
