"""RiskManager agent for Probably Fine Capital.

Pure Python — no LLM calls. Acts as the hard gate between the research desk
and trade execution. Every AnalystReport must pass all six gates to be approved.
"""

from __future__ import annotations

import config
from core.models import AnalystReport, FundState, RiskDecision
from utils.logger import system_logger as logger


class RiskManager:
    """Hard gate that filters AnalystReports against Python-enforced risk limits."""

    def evaluate(
        self,
        reports: list[AnalystReport],
        fund_state: FundState,
    ) -> list[RiskDecision]:
        """Evaluate each report against all risk gates in order.

        Gate 2 (fund drawdown) is checked once before the per-report loop.
        All other gates are checked per report. Any gate failure → veto.
        Hold signals are vetoed silently (no warning log) because they are expected.

        Args:
            reports:    Flat list from ResearchDesk.analyze().
            fund_state: Current fund snapshot with cash, positions, drawdown.

        Returns:
            One RiskDecision per input report, in the same order.
        """
        # Gate 2 evaluated once — if breached, every non-hold report is vetoed
        try:
            drawdown_breached = fund_state.drawdown_pct >= config.MAX_DRAWDOWN_PCT
        except Exception as exc:
            logger.error("RiskManager: could not read fund drawdown — vetoing all: %s", exc)
            drawdown_breached = True

        if drawdown_breached:
            logger.warning(
                "RiskManager: fund drawdown %.4f >= MAX_DRAWDOWN_PCT %.4f — halting all trades",
                fund_state.drawdown_pct,
                config.MAX_DRAWDOWN_PCT,
            )

        decisions: list[RiskDecision] = []
        for report in reports:
            try:
                decision = self._evaluate_one(report, fund_state, drawdown_breached)
            except Exception as exc:
                logger.error(
                    "RiskManager: unexpected exception evaluating %s — vetoing: %s",
                    report.ticker,
                    exc,
                )
                decision = RiskDecision(
                    approved=False,
                    modified_confidence=0.0,
                    veto_reason=f"internal error: {exc}",
                    original_report=report,
                )
            decisions.append(decision)

        approved = sum(1 for d in decisions if d.approved)
        vetoed = len(decisions) - approved
        logger.info(
            "Risk review complete: %d approved, %d vetoed",
            approved,
            vetoed,
        )
        return decisions

    def _evaluate_one(
        self,
        report: AnalystReport,
        fund_state: FundState,
        drawdown_breached: bool,
    ) -> RiskDecision:
        """Run all gates for one report. Returns the first failing gate's veto."""
        # Holds are expected — veto silently, no logger.warning
        if report.signal == "hold":
            return RiskDecision(
                approved=False,
                modified_confidence=0.0,
                veto_reason="hold signal — no action",
                original_report=report,
            )

        # Gate 1 — confidence threshold
        if report.confidence < config.MIN_CONFIDENCE:
            return self._veto(
                report,
                f"confidence {report.confidence:.2f} below minimum {config.MIN_CONFIDENCE:.2f}",
            )

        # Gate 2 — fund drawdown (pre-computed)
        if drawdown_breached:
            return self._veto(
                report,
                f"fund drawdown {fund_state.drawdown_pct:.2%} exceeds maximum",
            )

        # Gate 3 — max open positions (buy only)
        if (
            report.signal == "buy"
            and len(fund_state.positions) >= config.MAX_OPEN_POSITIONS
        ):
            return self._veto(
                report,
                f"max open positions {config.MAX_OPEN_POSITIONS} reached",
            )

        # Gate 4 — existing position in same ticker (buy only)
        if report.signal == "buy" and report.ticker in fund_state.positions:
            return self._veto(
                report,
                f"position already open for {report.ticker}",
            )

        # Gate 5 — no position to sell (sell only)
        if report.signal == "sell" and report.ticker not in fund_state.positions:
            return self._veto(
                report,
                f"no open position for {report.ticker} to sell",
            )

        # Gate 6 — trade size bounds
        estimated_size = fund_state.cash * config.MIN_CONFIDENCE
        if estimated_size < config.MIN_TRADE_SIZE_USD:
            return self._veto(
                report,
                f"estimated size ${estimated_size:.2f} below minimum",
            )

        return RiskDecision(
            approved=True,
            modified_confidence=report.confidence,
            veto_reason=None,
            original_report=report,
        )

    @staticmethod
    def _veto(report: AnalystReport, reason: str) -> RiskDecision:
        """Log and return a veto decision."""
        logger.warning(
            "RiskManager: vetoed %s %s — %s",
            report.ticker,
            report.signal,
            reason,
        )
        return RiskDecision(
            approved=False,
            modified_confidence=0.0,
            veto_reason=reason,
            original_report=report,
        )
