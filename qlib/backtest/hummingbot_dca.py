"""Standalone Qlib backtest entrypoint for pinned Hummingbot DCA semantics."""

from decimal import Decimal
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def simulate_hummingbot_dca(
    df: pd.DataFrame,
    prices: Sequence[Decimal],
    amounts_quote: Sequence[Decimal],
    take_profit: Optional[Decimal],
    stop_loss: Optional[Decimal],
    trade_cost: float,
    side="BUY",
    mode="MAKER",
    trailing_stop: Optional[object] = None,
    time_limit: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """Simulate Hummingbot's supported MAKER DCA semantics and return its ledger and close type.

    ``df`` must start at the signal bar and use float epoch seconds for both its
    index and ``timestamp`` column. This function is exported from
    ``qlib.backtest`` as ``backtest_hummingbot_dca`` for callers that need this
    execution path; it deliberately does not alter the generic Qlib backtest or
    Exchange order path. BUY/MAKER remains the backward-compatible default.
    """
    side = getattr(side, "value", side)
    mode = getattr(mode, "value", mode)
    if mode == "TAKER":
        raise NotImplementedError("Taker mode is not supported in DCAExecutorSimulator")
    if mode != "MAKER":
        raise ValueError("mode must be MAKER")
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be BUY or SELL")
    side_multiplier = 1 if side == "BUY" else -1
    trailing_activation = None
    trailing_delta = None
    if trailing_stop is not None:
        if isinstance(trailing_stop, Mapping):
            trailing_activation = trailing_stop.get("activation_price")
            trailing_delta = trailing_stop.get("trailing_delta")
        else:
            trailing_activation = getattr(trailing_stop, "activation_price", None)
            trailing_delta = getattr(trailing_stop, "trailing_delta", None)
        if trailing_activation is None or trailing_delta is None:
            raise ValueError("trailing_stop must define activation_price and trailing_delta")
        trailing_activation = (
            trailing_activation if isinstance(trailing_activation, Decimal) else Decimal(str(trailing_activation))
        )
        trailing_delta = trailing_delta if isinstance(trailing_delta, Decimal) else Decimal(str(trailing_delta))
        if (
            not trailing_activation.is_finite()
            or trailing_activation <= 0
            or not trailing_delta.is_finite()
            or trailing_delta <= 0
        ):
            raise ValueError("trailing_stop activation_price and trailing_delta must be finite and positive")

    required_columns = {"timestamp", "close", "low"}
    if side == "SELL" and stop_loss is not None:
        required_columns.add("high")
    if not required_columns.issubset(df.columns):
        raise ValueError(f"df must contain {sorted(required_columns)}")
    if df.empty or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("df must be non-empty with a sorted, unique timestamp index")
    index_values = df.index.to_numpy(dtype=float)
    timestamp_values = df["timestamp"].to_numpy(dtype=float)
    if not np.isfinite(index_values).all() or not np.array_equal(index_values, timestamp_values):
        raise ValueError("df index and timestamp column must contain matching epoch seconds")
    price_columns = ["close", "low"]
    if "high" in required_columns:
        price_columns.append("high")
    if not np.isfinite(df[price_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"{', '.join(price_columns)} must contain finite prices")

    prices = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in prices]
    amounts_quote = [value if isinstance(value, Decimal) else Decimal(str(value)) for value in amounts_quote]
    take_profit = (
        None
        if take_profit is None
        else (take_profit if isinstance(take_profit, Decimal) else Decimal(str(take_profit)))
    )
    stop_loss = None if stop_loss is None else stop_loss if isinstance(stop_loss, Decimal) else Decimal(str(stop_loss))
    if not prices or len(prices) != len(amounts_quote):
        raise ValueError("prices and amounts_quote must have the same non-zero length")
    if any(not value.is_finite() or value <= 0 for value in prices + amounts_quote):
        raise ValueError("prices and amounts_quote must be finite and positive")
    if take_profit is not None and (not take_profit.is_finite() or take_profit <= 0):
        raise ValueError("take_profit must be finite and positive")
    if stop_loss is not None and (not stop_loss.is_finite() or stop_loss <= 0):
        raise ValueError("stop_loss must be finite and positive")
    if not np.isfinite(trade_cost) or trade_cost < 0:
        raise ValueError("trade_cost must be finite and non-negative")
    if time_limit is not None and (not np.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and non-negative")

    last_timestamp = df["timestamp"].max()
    time_limit_timestamp = last_timestamp
    if time_limit:
        time_limit_timestamp = df["timestamp"].iloc[0] + time_limit
    df_filtered = df[:time_limit_timestamp].copy()
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
        entry_condition = df_filtered["close"] <= price if side == "BUY" else df_filtered["close"] >= price
        entry_timestamp = df_filtered.loc[entry_condition, "timestamp"].min()
        if pd.isna(entry_timestamp):
            break

        returns_df = df_filtered[entry_timestamp:]
        returns = returns_df["close"].pct_change().fillna(0)
        cumulative_returns = ((1 + returns).cumprod() - 1) * side_multiplier - 2 * trade_cost
        take_profit_timestamp = np.nan
        if take_profit is not None:
            take_profit_price = break_even_price * (1 + take_profit * side_multiplier)
            take_profit_condition = (
                returns_df["close"] >= take_profit_price if side == "BUY" else returns_df["close"] <= take_profit_price
            )
            take_profit_timestamp = returns_df.loc[take_profit_condition, "timestamp"].min()
        stop_loss_timestamp = np.nan
        trailing_stop_timestamp = np.nan
        next_order_timestamp = np.nan
        if trailing_activation is not None and trailing_delta is not None:
            trailing_activation_price = break_even_price * (1 + trailing_activation * side_multiplier)
            if side == "BUY":
                activated = returns_df["close"] >= trailing_activation_price
                if activated.any():
                    activated = activated.cumsum() > 0
                    with pd.option_context("mode.chained_assignment", None):
                        returns_df.loc[activated, "ts_trigger_price"] = (
                            returns_df.loc[activated, "close"] * float(1 - trailing_delta)
                        ).cummax()
                    trailing_stop_condition = returns_df["close"] <= returns_df["ts_trigger_price"]
                    trailing_stop_timestamp = returns_df.loc[trailing_stop_condition, "timestamp"].min()
            else:
                activated = returns_df["close"] <= trailing_activation_price
                if activated.any():
                    activated = activated.cumsum() > 0
                    with pd.option_context("mode.chained_assignment", None):
                        returns_df.loc[activated, "ts_trigger_price"] = (
                            returns_df.loc[activated, "close"] * float(1 + trailing_delta)
                        ).cummin()
                    trailing_stop_condition = returns_df["close"] >= returns_df["ts_trigger_price"]
                    trailing_stop_timestamp = returns_df.loc[trailing_stop_condition, "timestamp"].min()
        if level == len(prices) - 1:
            if stop_loss is not None:
                stop_loss_price = break_even_price * (1 - stop_loss * side_multiplier)
                stop_loss_column = "low" if side == "BUY" else "high"
                stop_loss_condition = (
                    returns_df[stop_loss_column] <= stop_loss_price
                    if side == "BUY"
                    else returns_df[stop_loss_column] >= stop_loss_price
                )
                stop_loss_timestamp = returns_df.loc[stop_loss_condition, "timestamp"].min()
        else:
            next_order_condition = (
                returns_df["close"] <= prices[level + 1] if side == "BUY" else returns_df["close"] >= prices[level + 1]
            )
            next_order_timestamp = returns_df.loc[next_order_condition, "timestamp"].min()

        close_timestamp = min(
            timestamp
            for timestamp in (
                take_profit_timestamp,
                stop_loss_timestamp,
                trailing_stop_timestamp,
                last_timestamp,
                next_order_timestamp,
            )
            if not pd.isna(timestamp)
        )
        if close_timestamp == take_profit_timestamp:
            close_type = "TAKE_PROFIT"
        elif close_timestamp == stop_loss_timestamp:
            close_type = "STOP_LOSS"
        elif close_timestamp == trailing_stop_timestamp:
            close_type = "TRAILING_STOP"
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
            df_filtered.loc[stage_entry_timestamp:, f"net_pnl_quote_{level}"] = (
                stage["cumulative_returns"] * stage["amount"]
            )
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
