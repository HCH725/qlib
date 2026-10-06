from decimal import Decimal

import pandas as pd

from qlib.backtest import backtest_hummingbot_dca


def test_qlib_backtest_dca_entrypoint_uses_each_stage_entry_timestamp():
    """The public Qlib DCA entrypoint matches Hummingbot's per-stage fill timing."""
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
    expected_average = float(
        sum(amounts_quote[level] * prices[level] for level in range(4)) / sum(amounts_quote[:4])
    )
    assert abs(ledger["current_position_average_price"].iloc[-1] - expected_average) < 1e-8

    assert ledger["filled_amount_quote_0"].tolist() == [1000.0] * 9
    assert ledger["filled_amount_quote_1"].tolist() == [0.0] + [1100.0] * 8
    assert ledger["filled_amount_quote_2"].tolist() == [0.0] * 5 + [1210.0] * 4
    assert ledger["filled_amount_quote_3"].tolist() == [0.0] * 5 + [1331.0] * 4
    assert ledger["filled_amount_quote"].iloc[0] == 1000.0
    assert ledger["filled_amount_quote"].iloc[1] == 2100.0
    assert ledger["filled_amount_quote"].iloc[5] == 4641.0
    assert ledger["filled_amount_quote"].iloc[-1] == 9282.0
    assert abs(ledger["net_pnl_quote"].iloc[-1] - 68.78431995) < 1e-8
    assert abs(ledger["cum_fees_quote"].iloc[-1] - 1.8564) < 1e-12


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
