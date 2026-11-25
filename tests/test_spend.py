import pytest

from dne.spend import BudgetExceeded, SpendLedger, cost_usd


def test_cost_opus():
    assert cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.00)


def test_record_persists_and_totals(tmp_path):
    p = tmp_path / "ledger.csv"
    led = SpendLedger(p)
    led.record("judge", "claude-opus-5-5", 100_000, 10_000)  # 0.40 + 0.20
    led.record("judge", "claude-sonnet-5-5", 100_000, 10_000)  # 0.20 + 0.10
    assert SpendLedger(p).total() == pytest.approx(0.90)


def test_check_blocks_at_cap(tmp_path):
    led = SpendLedger(tmp_path / "l.csv", cap_usd=20.0)
    led.record("seed", "claude-opus-5-5", 4_875_000, 0)  # $19.50
    with pytest.raises(BudgetExceeded):
        led.check(0.60)
    led.check(0.40)


def test_warn_once(tmp_path, capsys):
    led = SpendLedger(tmp_path / "l.csv", warn_usd=10.0)
    led.record("a", "claude-opus-5-5", 2_600_000, 0)  # 10.40
    led.record("b", "claude-opus-5-5", 250_000, 0)  # 11.40
    assert capsys.readouterr().out.count("passed $10") == 1


def test_unknown_model():
    with pytest.raises(KeyError):
        cost_usd("gpt-x", 1, 1)


@pytest.mark.parametrize("model", ["claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5-2"])
def test_fallback_models_priced(model):
    # server-side refusal fallbacks may serve the call on another model; the ledger must still price it
    assert cost_usd(model, 1_000_000, 0) > 0
