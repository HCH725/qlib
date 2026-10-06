"""Standalone Qlib backtest entrypoint for pinned Hummingbot DCA semantics."""

from decimal import Decimal
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def simulate_hummingbot_dca(
    df: pd.DataFrame,
    prices: Sequence[Decimal],
    amounts_quote: Sequence[Decimal],
    take_profit: Decimal,
    stop_loss: Decimal,
    trade_cost: float,
) -> Tuple[pd.DataFrame, str]:
    """Simulate the supported BUY/MAKER DCA slice and return its ledger and close type.

    ``df`` must start at the signal bar and use float epoch seconds for both its
    index and ``timestamp`` column. This function is exported from
    ``qlib.backtest`` as ``backtest_hummingbot_dca`` for callers that need this
    execution path; it deliberately does not alter the generic Qlib backtest or
    Exchange order path. The supported case has no trailing stop or time limit.
    """
    required_columns = {"timestamp", "close", "low"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(required_columns)}")
    if df.empty or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("df must be non-empty with a sorted, unique timestamp index")
    index_values = df.index.to_numpy(dtype=float)
    timestamp_values = df["timestamp"].to_numpy(dtype=float)
    if not np.isfinite(index_values).all() or not np.array_equal(index_values, timestamp_values):
        raise ValueError("df index and timestamp column must contain matching epoch seconds")
    if not np.isfinite(df[["close", "low"]].to_numpy(dtype=float)).all():
        raise ValueError("close and low must contain finite prices")

    prices = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in prices]
    amounts_quote = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in amounts_quote]
    take_profit = take_profit if isinstance(take_profit, Decimal) else Decimal(str(take_profit))
    stop_loss = stop_loss if isinstance(stop_loss, Decimal) else Decimal(str(stop_loss))
    if not prices or len(prices) != len(amounts_quote):
        raise ValueError("prices and amounts_quote must have the same non-zero length")
    if any(not value.is_finite() or value <= 0 for value in prices + amounts_quote):
        raise ValueError("prices and amounts_quote must be finite and positive")
    if not take_profit.is_finite() or take_profit <= 0 or not stop_loss.is_finite() or stop_loss <= 0:
        raise ValueError("take_profit and stop_loss must be finite and positive")
    if not np.isfinite(trade_cost) or trade_cost < 0:
        raise ValueError("trade_cost must be finite and non-negative")

    last_timestamp = df["timestamp"].max()
    df_filtered = df[:last_timestamp].copy()
    df_filtered["net_pnl_pct"] = 0.0
    df_filtered["net_pnl_quote"] = 0.0
    df_filtered["cum_fees_quote"] = 0.0
    df_filtered["filled_amount_quote"] = 0.0
    df_filtered["current_position_average_price"] = float(prices[0])

    potential_stages = []
    entry_timestamp = np.nan
    for level, (price, amount) in enumerate(zip(prices, amounts_quote)):
        total_amount = sum(amounts_quote[: level + 1])
        total_quote = sum(amounts_quote[i] * prices[i] for i in range(level + 1))
        break_even_price = total_quote / total_amount
        entry_timestamp = df_filtered.loc[df_filtered["close"] <= price, "timestamp"].min()
        if pd.isna(entry_timestamp):
            break

        returns_df = df_filtered[entry_timestamp:]
        returns = returns_df["close"].pct_change().fillna(0)
        cumulative_returns = ((1 + returns).cumprod() - 1) - 2 * trade_cost
        take_profit_price = break_even_price * (1 + take_profit)
        take_profit_timestamp = returns_df.loc[returns_df["close"] >= take_profit_price, "timestamp"].min()
        stop_loss_timestamp = np.nan
        next_order_timestamp = np.nan
        if level == len(prices) - 1:
            stop_loss_price = break_even_price * (1 - stop_loss)
            stop_loss_timestamp = returns_df.loc[returns_df["low"] <= stop_loss_price, "timestamp"].min()
        else:
            next_order_timestamp = returns_df.loc[returns_df["close"] <= prices[level + 1], "timestamp"].min()

        close_timestamp = min(
            timestamp
            for timestamp in (
                take_profit_timestamp,
                stop_loss_timestamp,
                last_timestamp,
                next_order_timestamp,
            )
            if not pd.isna(timestamp)
        )
        if close_timestamp == take_profit_timestamp:
            close_type = "TAKE_PROFIT"
        elif close_timestamp == stop_loss_timestamp:
            close_type = "STOP_LOSS"
        elif close_timestamp == next_order_timestamp:
            close_type = None
        else:
            close_type = "TIME_LIMIT"

        df_filtered[f"filled_amount_quote_{level}"] = 0.0
        df_filtered[f"net_pnl_quote_{level}"] = 0.0
        potential_stages.append(
            {
                "entry_timestamp": entry_timestamp,
                "amount": float(amount),
                "break_even_price": float(break_even_price),
                "close_timestamp": close_timestamp,
                "close_type": close_type,
                "cumulative_returns": cumulative_returns,
            }
        )

    if not potential_stages:
        return df_filtered, "TIME_LIMIT"

    close_type = None
    for level, stage in enumerate(potential_stages):
        stage_entry_timestamp = stage["entry_timestamp"]
        if not pd.isna(stage_entry_timestamp):
            df_filtered.loc[stage_entry_timestamp:, f"filled_amount_quote_{level}"] = stage["amount"]
            df_filtered.loc[stage_entry_timestamp:, f"net_pnl_quote_{level}"] = stage["cumulative_returns"] * stage["amount"]
            df_filtered.loc[stage_entry_timestamp:, "current_position_average_price"] = stage["break_even_price"]
        if stage["close_type"] is not None:
            close_type = stage["close_type"]
            last_timestamp = stage["close_timestamp"]
            break

    df_filtered = df_filtered[:last_timestamp].copy()
    fill_columns = [f"filled_amount_quote_{level}" for level in range(len(potential_stages))]
    pnl_columns = [f"net_pnl_quote_{level}" for level in range(len(potential_stages))]
    df_filtered["filled_amount_quote"] = df_filtered[fill_columns].sum(axis=1)
    df_filtered["net_pnl_quote"] = df_filtered[pnl_columns].sum(axis=1)
    df_filtered["cum_fees_quote"] = 2 * trade_cost * df_filtered["filled_amount_quote"]
    filled = df_filtered["filled_amount_quote"] > 0
    df_filtered.loc[filled, "net_pnl_pct"] = (
        df_filtered.loc[filled, "net_pnl_quote"] / df_filtered.loc[filled, "filled_amount_quote"]
    )
    df_filtered.loc[df_filtered.index[-1], "filled_amount_quote"] *= 2

    return df_filtered, close_type or "FAILED"
