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

# Captures one unquoted scalar value slot: ": <value>" up to the next , } or ]
_VALUE_SLOT_RE = re.compile(r'(:\s*)([^"\[\{\s][^,}\]]*?)(\s*)(?=[,}\]])')
# A value safe to evaluate: only digits, spaces, . and + - * / ( ) — nothing else
_SAFE_ARITH_RE = re.compile(r"[\d\s+\-*/().]+")


def _eval_arithmetic_exprs(text: str) -> str:
    """Replace arithmetic expressions in JSON value slots with their numeric result.

    The model sometimes emits values like ``0.5 + 0.2``, ``.6 * 0.8`` or
    ``(0.5 + 0.2) / 2`` where a plain number is required. Each non-string,
    non-object value slot is inspected: already-valid JSON scalars are left
    untouched; anything composed solely of digits, spaces and + - * / . ( )
    is evaluated and substituted. eval() runs with no builtins and the input
    is char-restricted, so no names or calls are reachable.
    """
    def _replacer(m: re.Match) -> str:
        prefix, raw, suffix = m.group(1), m.group(2).strip(), m.group(3)
        if raw in ("true", "false", "null"):
            return m.group(0)
        try:
            json.loads(raw)  # already a valid number — leave it alone
            return m.group(0)
        except ValueError:
            pass
        if _SAFE_ARITH_RE.fullmatch(raw):
            try:
                val = eval(raw, {"__builtins__": {}}, {})  # noqa: S307
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return f"{prefix}{val!r}{suffix}"
            except Exception:
                pass
        return m.group(0)
    return _VALUE_SLOT_RE.sub(_replacer, text)


def _extract_first_json_object(text: str) -> str | None:
    """Return the first complete balanced ``{...}`` object in ``text``.

    Walks the string tracking brace depth while respecting string literals
    and backslash escapes, so trailing prose or extra objects emitted after
    the first one are discarded (fixes "Extra data" parse failures). Empty
    ``{}`` objects are skipped — no response model has zero required fields,
    so an empty object is always a placeholder the model emitted before the
    real one. If an object is truncated (depth never returns to zero), the
    missing closing braces are appended for a best-effort parse. Returns
    None when no non-empty object is found.
    """
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if candidate[1:-1].strip():
                        return candidate
                    search_from = i + 1  # empty {} placeholder — keep scanning
                    break
        else:
            if depth > 0:
                return text[start:] + "}" * depth
            return None


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
                    response_format={
                        "type": "json_object",
                        "schema": response_model.model_json_schema(),
                    },
                )
                await asyncio.sleep(2.0)
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

            # Step 3: extract the first balanced {...} object (discards trailing
            # prose / extra objects, repairs truncated ones)
            extracted = _extract_first_json_object(text)
            if extracted is None:
                logger.error(
                    "call_llm: no JSON object found on attempt %d for %s — text: %.200s",
                    attempt + 1,
                    response_model.__name__,
                    text,
                )
                return None

            # Step 4: clean (strip comments, trailing commas, arithmetic exprs) then parse
            cleaned = re.sub(r"//.*$", "", extracted, flags=re.MULTILINE)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            cleaned = _eval_arithmetic_exprs(cleaned)
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
                        parsed = ast.literal_eval(cleaned)
                    except (ValueError, SyntaxError) as exc3:
                        logger.warning(
                            "call_llm: ast.literal_eval failed on attempt %d for %s: %s",
                            attempt + 1,
                            response_model.__name__,
                            exc3,
                        )
                        raise json.JSONDecodeError(str(exc3), extracted, 0) from exc3

            return response_model.model_validate(parsed)
        except ValidationError as exc:
            logger.warning(
                "call_llm: Pydantic validation failed on attempt %d for %s: %s — parsed: %s",
                attempt + 1,
                response_model.__name__,
                exc,
                parsed,
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                "call_llm: JSON error on attempt %d for %s: %s",
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
