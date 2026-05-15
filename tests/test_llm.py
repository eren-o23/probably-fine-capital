"""Tests for utils/llm.py — semaphore concurrency cap and 429 retry logic."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from utils.llm import call_llm


class _TestModel(BaseModel):
    value: str


def _ok_response() -> MagicMock:
    msg = MagicMock()
    msg.content = '{"value": "ok"}'
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError(
        "Too Many Requests",
        response=httpx.Response(
            status_code=429,
            request=httpx.Request("POST", "https://api.fireworks.ai"),
        ),
        body={},
    )


# ---------------------------------------------------------------------------
# Semaphore concurrency cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_limits_concurrency_to_five() -> None:
    """10 concurrent call_llm calls never exceed 5 simultaneous HTTP requests."""
    concurrent = 0
    max_concurrent = 0

    async def tracked_create(**kwargs):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return _ok_response()

    mock_client = MagicMock()
    mock_client.chat.completions.create = tracked_create

    with patch("utils.llm._get_client", return_value=mock_client):
        await asyncio.gather(*[call_llm("prompt", _TestModel) for _ in range(10)])

    assert max_concurrent <= 5


# ---------------------------------------------------------------------------
# 429 retry with backoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_429_retries_and_succeeds_on_second_attempt() -> None:
    """First call raises 429, second succeeds → returns result, sleeps 2s."""
    create_mock = AsyncMock(side_effect=[_rate_limit_error(), _ok_response()])
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_mock

    with patch("utils.llm._get_client", return_value=mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await call_llm("prompt", _TestModel)

    assert result == _TestModel(value="ok")
    mock_sleep.assert_awaited_once_with(2)  # 2 ** 1 = 2s on first retry


@pytest.mark.asyncio
async def test_429_three_times_returns_none() -> None:
    """All 3 attempts raise 429 → returns None, sleeps twice (not after last attempt)."""
    create_mock = AsyncMock(side_effect=[
        _rate_limit_error(), _rate_limit_error(), _rate_limit_error(),
    ])
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_mock

    with patch("utils.llm._get_client", return_value=mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await call_llm("prompt", _TestModel)

    assert result is None
    assert create_mock.await_count == 3
    assert mock_sleep.await_count == 2  # sleep after attempt 1 (2s) and 2 (4s), not 3


# ---------------------------------------------------------------------------
# Non-429 error — fail immediately, no retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_429_error_fails_immediately_no_retry() -> None:
    """A non-429 exception returns None after a single attempt — no backoff sleep."""
    create_mock = AsyncMock(side_effect=ConnectionError("network down"))
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_mock

    with patch("utils.llm._get_client", return_value=mock_client), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await call_llm("prompt", _TestModel)

    assert result is None
    assert create_mock.await_count == 1
    mock_sleep.assert_not_awaited()
