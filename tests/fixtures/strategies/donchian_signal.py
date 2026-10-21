#!/usr/bin/env python3
"""Frozen strategy artifact: Donchian channel breakout (S01509 proof fixture).

This is one ordinary strategy artifact, not an engine rule. It was moved out of
``qlib/backtest/research_cell.py`` when that file became a coverage + shared
execution shell, so the engine no longer names this strategy, its parameters or
its indicators. Nothing here is imported by the engine at import time.

Rule, unchanged from the pinned proof (completed-bar decision; the prior-N
channel excludes the current bar):

- long from flat when ``close`` breaks above the prior ``entry_len`` highs,
  only inside the frozen window;
- back to flat when ``close`` breaks below the prior ``exit_len`` lows;
- one entry per run; ``entry_ts`` is the first entry transition.

The parameters come from the frozen record through
``effective_config.json -> strategy.params``; this file invents no default.
"""


def signal(*, high, low, close, volume, times, params, window_start, window_end):
    """Return (rows, entry_ts) over the aligned frozen-provider bars."""
    entry_len = params.get("entry_len")
    exit_len = params.get("exit_len")
    for name, value in (("entry_len", entry_len), ("exit_len", exit_len)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("donchian artifact requires a positive integer %s" % name)
    state = False
    entry_ts = None
    rows = []
    for i in range(len(close)):
        prior_high = max(high[i - entry_len:i]) if i >= entry_len else None
        prior_low = min(low[i - exit_len:i]) if i >= exit_len else None
        in_window = window_start <= times[i] <= window_end
        break_up = prior_high is not None and close[i] > prior_high
        break_down = prior_low is not None and close[i] < prior_low
        if break_up and break_down:
            raise ValueError("channel collision at bar %d" % i)
        entry = exit_ = False
        if not state and in_window and break_up:
            state, entry, entry_ts = True, True, times[i]
        elif state and break_down:
            state, exit_ = False, True
        rows.append(
            {
                "datetime": times[i],
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "prior_high": None if prior_high is None else float(prior_high),
                "prior_low": None if prior_low is None else float(prior_low),
                "in_window": int(in_window),
                "break_up": int(break_up),
                "break_down": int(break_down),
                "state": int(state),
                "entry_transition": int(entry),
                "exit_transition": int(exit_),
            }
        )
    return rows, entry_ts
