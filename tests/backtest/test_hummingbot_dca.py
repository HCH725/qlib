from decimal import Decimal

import pandas as pd

from qlib.backtest import backtest_hummingbot_dca


def test_qlib_backtest_dca_entrypoint_preserves_pinned_fill_ledger():
    """The public Qlib DCA entrypoint preserves the pinned golden executor ledger."""
    timestamps = [1642420800.0 + 14400.0 * index for index in range(9)]
    closes = [142.07, 139.85, 139.74, 139.23, 140.15, 137.60, 136.58, 136.96, 141.19]
    candles = pd.DataFrame(
        {"timestamp": timestamps, "close": closes, "low": closes},
        index=pd.Index(timestamps, name="timestamp"),
    )
    p0 = Decimal("142.07")
    prices = [p0 * (Decimal("1") - Decimal("0.01") * level) for level in range(11)]
    amounts_quote = [Decimal("1000") * Decimal("1.1") ** level for level in range(11)]

    ledger, close_type = backtest_hummingbot_dca(
        candles,
        prices,
        amounts_quote,
        take_profit=Decimal("0.01"),
        stop_loss=Decimal("0.10"),
        trade_cost=0.0002,
    )

    assert close_type == "TAKE_PROFIT"
    assert len(ledger) == 9
    assert ledger.index[-1] == 1642536000.0
    assert ledger["close"].iloc[-1] == 141.19
    assert ledger["current_position_average_price"].iloc[-1] == 142.07
    assert (ledger["filled_amount_quote"] == 0.0).all()
    assert (ledger["net_pnl_quote"] == 0.0).all()
    assert (ledger["cum_fees_quote"] == 0.0).all()
    for level in range(4):
        assert (ledger[f"filled_amount_quote_{level}"] == 0.0).all()


def test_decimal_threshold_comparison_matches_pinned_simulator():
    timestamps = [1.0, 2.0, 3.0]
    closes = [0.1, 0.1, 0.11]
    candles = pd.DataFrame(
        {"timestamp": timestamps, "close": closes, "low": closes},
        index=pd.Index(timestamps, name="timestamp"),
    )

    ledger, close_type = backtest_hummingbot_dca(
        candles,
        [Decimal("0.1")],
        [Decimal("100")],
        take_profit=Decimal("0.01"),
        stop_loss=Decimal("0.10"),
        trade_cost=0.0002,
    )

    assert close_type == "TIME_LIMIT"
    assert ledger["filled_amount_quote"].tolist() == [0.0, 0.0, 0.0]
