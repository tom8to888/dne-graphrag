"""API spend ledger for the Anthropic judge, with a hard cap ($20; warning at $10)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

# USD per 1M (input, output) tokens — Anthropic price list, 2026-11-03.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    # models a server-side refusal fallback may hand the call to
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-fable-5-2": (8.00, 40.00),
}
PRICES_AS_OF = "2026-11-03"
_FIELDS = ["ts", "stage", "model", "input_tokens", "output_tokens", "usd"]


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


class BudgetExceeded(Exception):
    pass


class SpendLedger:
    def __init__(self, path: Path, cap_usd: float = 20.0, warn_usd: float = 10.0):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.warn_usd = warn_usd
        self._warned = False
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(_FIELDS)

    def total(self) -> float:
        with self.path.open(newline="") as f:
            return sum(float(r["usd"]) for r in csv.DictReader(f))

    def record(self, stage: str, model: str, input_tokens: int, output_tokens: int) -> float:
        before = self.total()
        usd = cost_usd(model, input_tokens, output_tokens)
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), stage, model,
                                    input_tokens, output_tokens, f"{usd:.6f}"])
        total = before + usd
        if not self._warned and before <= self.warn_usd < total:
            print(f"⚠ Claude spend ${total:.2f} passed ${self.warn_usd:.0f}")
            self._warned = True
        return total

    def check(self, next_estimate_usd: float) -> None:
        total = self.total()
        if total + next_estimate_usd > self.cap_usd:
            raise BudgetExceeded(
                f"ledger ${total:.2f} + next ≈${next_estimate_usd:.2f} would pass the ${self.cap_usd:.0f} cap")
