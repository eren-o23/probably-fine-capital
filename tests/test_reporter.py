"""Tests for Reporter agent.

call_llm is mocked throughout — no Fireworks API calls.
All file writes go to a per-test tmp_path so logs/reports/ stays clean.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agents.reporter import (
    Reporter,
    _NarrativeResponse,
    _ThreadResponse,
    _truncate_at_word_boundary,
)
from core.models import FundState, Position, TradeInstruction


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_position(ticker: str = "AAPLx/USD", entry: float = 150.0, current: float = 165.0) -> Position:
    return Position(
        ticker=ticker,
        size_usd=300.0,
        quantity=2.0,
        entry_price=entry,
        current_price=current,
        stop_loss_price=entry * 0.95,
        opened_at=datetime.now(timezone.utc),
    )


def _make_fund_state(with_positions: bool = True) -> FundState:
    positions = {}
    if with_positions:
        positions["AAPLx/USD"] = _make_position()
    return FundState(
        cash=8_000.0,
        starting_cash=10_000.0,
        peak_value=11_000.0,
        positions=positions,
    )


def _make_instruction(action: str = "buy", ticker: str = "AAPLx") -> TradeInstruction:
    return TradeInstruction(
        action=action,  # type: ignore[arg-type]
        ticker=ticker,
        size_usd=200.0,
        rationale="strong momentum signal",
    )


def _make_narrative() -> _NarrativeResponse:
    return _NarrativeResponse(
        headline="AI fund up 5% on tech rally",
        performance_summary="Portfolio gained 5% led by AAPL and NVDA. Cash deployed at 80%.",
        trade_recap="Bought AAPL on momentum breakout. Held NVDA through volatility.",
        market_observations="Mega-cap tech leading. Momentum strong, sentiment cautious.",
        outlook="Expect continued strength next session if SPY holds 500.",
        risk_flags="None",
    )


def _make_thread() -> _ThreadResponse:
    return _ThreadResponse(tweets=[
        "1/ Probably Fine Capital had a great day.",
        "2/ Portfolio up 5%, mostly tech.",
        "3/ Bought AAPL, held NVDA.",
        "4/ Outlook positive if SPY holds.",
        "5/ #ProbablyFineCapital #xStocks",
    ])


@pytest.fixture
def reporter(tmp_path: Path) -> Reporter:
    """Reporter with reports_dir redirected to a per-test tmp path."""
    r = Reporter()
    r.reports_dir = tmp_path / "reports"
    r.reports_dir.mkdir(parents=True, exist_ok=True)
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSuccessfulRun:
    async def test_returns_filepath_string(self, reporter: Reporter):
        with patch(
            "agents.reporter.call_llm",
            new_callable=AsyncMock,
            side_effect=[_make_narrative(), _make_thread()],
        ):
            filepath = await reporter.run(
                _make_fund_state(), [_make_instruction()], cycle_count=10, elapsed_seconds=3600.0
            )

        assert isinstance(filepath, str)
        assert filepath.endswith(".md")
        assert Path(filepath).exists()

    async def test_full_report_contains_all_narrative_sections(self, reporter: Reporter):
        narrative = _make_narrative()
        with patch(
            "agents.reporter.call_llm",
            new_callable=AsyncMock,
            side_effect=[narrative, _make_thread()],
        ):
            filepath = await reporter.run(
                _make_fund_state(), [_make_instruction()], cycle_count=5, elapsed_seconds=7200.0
            )

        content = Path(filepath).read_text(encoding="utf-8")
        assert narrative.headline in content
        assert narrative.performance_summary in content
        assert narrative.trade_recap in content
        assert narrative.market_observations in content
        assert narrative.outlook in content
        assert narrative.risk_flags in content
        assert "## X Thread Draft" in content

    async def test_filename_uses_utc_timestamp_format(self, reporter: Reporter):
        with patch(
            "agents.reporter.call_llm",
            new_callable=AsyncMock,
            side_effect=[_make_narrative(), _make_thread()],
        ):
            filepath = await reporter.run(
                _make_fund_state(), [], cycle_count=1, elapsed_seconds=60.0
            )

        name = Path(filepath).name
        assert name.startswith("report_")
        assert name.endswith(".md")
        # report_YYYYMMDD_HHMMSS.md → 7 + 15 + 3 = 25 chars
        assert len(name) == len("report_20260514_180000.md")


class TestNarrativeFailure:
    async def test_minimal_report_written_when_narrative_returns_none(self, reporter: Reporter):
        with patch("agents.reporter.call_llm", new_callable=AsyncMock, return_value=None):
            filepath = await reporter.run(
                _make_fund_state(), [_make_instruction()], cycle_count=3, elapsed_seconds=1800.0
            )

        assert Path(filepath).exists()
        content = Path(filepath).read_text(encoding="utf-8")
        assert "raw fallback" in content
        assert "$8,000.00" in content  # raw cash value
        assert "AAPLx/USD" in content  # open position rendered

    async def test_minimal_report_excludes_x_section(self, reporter: Reporter):
        with patch("agents.reporter.call_llm", new_callable=AsyncMock, return_value=None):
            filepath = await reporter.run(
                _make_fund_state(), [], cycle_count=1, elapsed_seconds=60.0
            )

        content = Path(filepath).read_text(encoding="utf-8")
        assert "X Thread Draft" not in content

    async def test_run_never_raises_on_narrative_exception(self, reporter: Reporter):
        """Even if call_llm raises unexpectedly, run() must return a filepath."""
        with patch("agents.reporter.call_llm", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            filepath = await reporter.run(
                _make_fund_state(), [], cycle_count=1, elapsed_seconds=60.0
            )

        assert Path(filepath).exists()


class TestThreadFailure:
    async def test_report_written_without_x_section_when_thread_fails(self, reporter: Reporter):
        with patch(
            "agents.reporter.call_llm",
            new_callable=AsyncMock,
            side_effect=[_make_narrative(), None],
        ):
            filepath = await reporter.run(
                _make_fund_state(), [_make_instruction()], cycle_count=4, elapsed_seconds=14400.0
            )

        content = Path(filepath).read_text(encoding="utf-8")
        # Narrative still present
        assert _make_narrative().headline in content
        # X section absent
        assert "X Thread Draft" not in content


class TestTweetLength:
    def test_truncate_short_string_unchanged(self):
        assert _truncate_at_word_boundary("hello world", 280) == "hello world"

    def test_truncate_exact_max_unchanged(self):
        text = "a" * 280
        assert _truncate_at_word_boundary(text, 280) == text

    def test_truncate_long_string_at_word_boundary(self):
        text = "word " * 100  # 500 chars
        result = _truncate_at_word_boundary(text, 280)
        assert len(result) <= 280
        assert result.endswith("…")
        # No partial words just before the ellipsis
        assert " " in result[-20:] or result[:-1].endswith("word")

    def test_truncate_no_space_falls_back_to_hard_cut(self):
        text = "x" * 500
        result = _truncate_at_word_boundary(text, 280)
        assert len(result) <= 280
        assert result.endswith("…")

    async def test_all_tweets_under_280_chars_in_written_report(self, reporter: Reporter):
        """LLM returns over-length tweets — verify they're truncated before write."""
        long_thread = _ThreadResponse(tweets=[
            "x " * 200,  # 400 chars
            "y " * 200,
            "z " * 200,
            "a " * 200,
            "b " * 200,
        ])
        with patch(
            "agents.reporter.call_llm",
            new_callable=AsyncMock,
            side_effect=[_make_narrative(), long_thread],
        ):
            filepath = await reporter.run(
                _make_fund_state(), [_make_instruction()], cycle_count=2, elapsed_seconds=600.0
            )

        content = Path(filepath).read_text(encoding="utf-8")
        # Each tweet appears in its own numbered markdown line "N. <tweet>"
        # Find the X Thread Draft section
        section_idx = content.index("## X Thread Draft")
        thread_lines = [
            line for line in content[section_idx:].splitlines()
            if line and line[0].isdigit() and ". " in line
        ]
        assert len(thread_lines) == 5
        for line in thread_lines:
            # Strip "N. " prefix
            tweet = line.split(". ", 1)[1]
            assert len(tweet) <= 280, f"tweet too long ({len(tweet)} chars): {tweet[:40]}..."


class TestPostToX:
    async def test_post_to_x_returns_false_when_disabled(self, reporter: Reporter):
        with patch.object(__import__("agents.reporter", fromlist=["config"]).config, "X_ENABLED", False):
            result = await reporter.post_to_x(["tweet 1", "tweet 2"])

        assert result is False

    async def test_post_to_x_returns_false_when_enabled_but_stubbed(self, reporter: Reporter):
        """Even with X_ENABLED=true the stub returns False — wiring is TODO."""
        with patch.object(__import__("agents.reporter", fromlist=["config"]).config, "X_ENABLED", True):
            result = await reporter.post_to_x(["tweet 1"])

        assert result is False
