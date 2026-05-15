"""Tests for Reporter.post_to_x() — tweepy integration.

All tests mock tweepy.Client so the real X API is never called.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

import config
from agents.reporter import Reporter


_GOOD_ENV = {
    "X_CONSUMER_KEY": "ck",
    "X_CONSUMER_SECRET": "cs",
    "X_ACCESS_TOKEN": "at",
    "X_ACCESS_TOKEN_SECRET": "ats",
}

_MISSING_ENV = {
    "X_CONSUMER_KEY": "",
    "X_CONSUMER_SECRET": "",
    "X_ACCESS_TOKEN": "",
    "X_ACCESS_TOKEN_SECRET": "",
}


@pytest.fixture()
def reporter(tmp_path: Path) -> Reporter:
    """Reporter with reports_dir redirected to a per-test tmp path."""
    r = Reporter()
    r.reports_dir = tmp_path / "reports"
    r.reports_dir.mkdir(parents=True, exist_ok=True)
    return r


# ---------------------------------------------------------------------------
# X disabled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_x_disabled_returns_false_no_tweepy_call(reporter: Reporter) -> None:
    """X_ENABLED=False → False immediately, tweepy.Client never instantiated."""
    with patch.object(config, "X_ENABLED", False):
        with patch("agents.reporter.tweepy") as mock_tweepy:
            result = await reporter.post_to_x(["hello"])
    assert result is False
    mock_tweepy.Client.assert_not_called()


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_credentials_returns_false(reporter: Reporter) -> None:
    """All four credentials empty → False, tweepy.Client never instantiated."""
    with patch.object(config, "X_ENABLED", True):
        with patch.dict("os.environ", _MISSING_ENV):
            with patch("agents.reporter.tweepy") as mock_tweepy:
                result = await reporter.post_to_x(["hello"])
    assert result is False
    mock_tweepy.Client.assert_not_called()


# ---------------------------------------------------------------------------
# tweepy raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tweepy_raises_on_first_tweet_returns_false(reporter: Reporter) -> None:
    """Exception from create_tweet → returns False."""
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = Exception("rate limit")

    with patch.object(config, "X_ENABLED", True):
        with patch.dict("os.environ", _GOOD_ENV):
            with patch("agents.reporter.tweepy") as mock_tweepy:
                mock_tweepy.Client.return_value = mock_client
                result = await reporter.post_to_x(["tweet 1"])

    assert result is False


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_success_returns_true_and_calls_create_tweet(reporter: Reporter) -> None:
    """All tweets sent → True, create_tweet called once per tweet."""
    tweets = ["tweet 1", "tweet 2", "tweet 3"]
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = [
        MagicMock(data={"id": 100}),
        MagicMock(data={"id": 101}),
        MagicMock(data={"id": 102}),
    ]

    with patch.object(config, "X_ENABLED", True):
        with patch.dict("os.environ", _GOOD_ENV):
            with patch("agents.reporter.tweepy") as mock_tweepy:
                mock_tweepy.Client.return_value = mock_client
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = await reporter.post_to_x(tweets)

    assert result is True
    assert mock_client.create_tweet.call_count == len(tweets)


# ---------------------------------------------------------------------------
# Reply chain
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reply_chain_wired_correctly(reporter: Reporter) -> None:
    """First tweet: no in_reply_to_tweet_id. Each subsequent one replies to the previous."""
    tweets = ["tweet 1", "tweet 2", "tweet 3"]
    mock_client = MagicMock()
    mock_client.create_tweet.side_effect = [
        MagicMock(data={"id": 10}),
        MagicMock(data={"id": 11}),
        MagicMock(data={"id": 12}),
    ]

    with patch.object(config, "X_ENABLED", True):
        with patch.dict("os.environ", _GOOD_ENV):
            with patch("agents.reporter.tweepy") as mock_tweepy:
                mock_tweepy.Client.return_value = mock_client
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    await reporter.post_to_x(tweets)

    calls = mock_client.create_tweet.call_args_list
    assert calls[0] == call(text="tweet 1")
    assert calls[1] == call(text="tweet 2", in_reply_to_tweet_id=10)
    assert calls[2] == call(text="tweet 3", in_reply_to_tweet_id=11)
