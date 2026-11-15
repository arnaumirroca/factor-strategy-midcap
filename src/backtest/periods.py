"""Holding-period returns between trade dates (Design Doc 7.2 and 10.2).

The signal is computed at month-end F. Orders are placed on the k-th trading day of
the next month (k = 2 in the design) and held until the k-th trading day of the month
after. So a stock's return for formation F runs from that first trade date to the next,
which avoids trading at the same close the signal was computed on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DELISTING_RETURN_FOR_CAUSE = -0.30
WITHHOLDING = 0.15  # US dividend withholding for a Spanish resident with a W-8BEN


def month_end(d) -> pd.Timestamp:
    return (pd.Timestamp(d) + pd.offsets.MonthEnd(0)).normalize()


def trade_dates(calendar: pd.DatetimeIndex, k: int) -> pd.Series:
    """The k-th trading day of each month (the last one if the month is shorter), keyed by month-end."""
    cal = pd.DatetimeIndex(calendar).sort_values().unique()
    keys = (cal + pd.offsets.MonthEnd(0)).normalize()
    s = pd.Series(cal, index=keys)
    out = s.groupby(level=0).apply(lambda x: x.iloc[min(k, len(x)) - 1])
    out.index = pd.DatetimeIndex(out.index).astype("datetime64[ns]")
    return out.astype("datetime64[ns]")


def holding_windows(formations, tdates: pd.Series) -> pd.DataFrame:
    """Start and end trade dates for each formation month-end F (F+1 and F+2 trade dates)."""
    rows = []
    for F in formations:
        m1, m2 = month_end(F + pd.offsets.MonthEnd(1)), month_end(F + pd.offsets.MonthEnd(2))
        if m1 in tdates.index and m2 in tdates.index:
            rows.append((pd.Timestamp(F), tdates[m1], tdates[m2]))
    return pd.DataFrame(rows, columns=["formation", "start", "end"]).set_index("formation")


def period_returns(adj: pd.DataFrame, close: pd.DataFrame, start, end, for_cause_last: pd.Series | None = None,
                   withholding: float = WITHHOLDING) -> pd.Series:
    """Net total return from `start` to `end` for every stock that trades on `start`.

    adj:   dividend- and split-adjusted closes (days x tickers)
    close: split-adjusted closes, without dividends, for the dividend part
    A stock that stops trading during the period keeps its last price (cash out). If it
    was delisted for cause (bankruptcy, regulatory), a further -30% is applied.
    Dividends are taxed at `withholding`.
    """
    a0 = adj.loc[:start].iloc[-1]
    c0 = close.loc[:start].iloc[-1]
    traded_at_start = adj.loc[:start].index[-1] == pd.Timestamp(start)
    ok = a0.notna() & traded_at_start
    window_adj = adj.loc[start:end].ffill().iloc[-1]
    window_close = close.loc[start:end].ffill().iloc[-1]
    tr = window_adj / a0 - 1.0
    pr = window_close / c0 - 1.0
    dividend = (tr - pr).clip(lower=0.0)
    net = (tr - withholding * dividend)[ok]
    if for_cause_last is not None and len(for_cause_last):
        hit = for_cause_last[(for_cause_last > pd.Timestamp(start)) & (for_cause_last <= pd.Timestamp(end))]
        hit = hit.index.intersection(net.index)
        net.loc[hit] = (1 + net.loc[hit]) * (1 + DELISTING_RETURN_FOR_CAUSE) - 1
    return net.dropna()


def fx_at(fx: pd.Series, dates) -> pd.Series:
    """USD per EUR on each date, using the latest ECB rate on or before it."""
    s = fx.sort_index()
    idx = s.index.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
    if (idx < 0).any():
        raise ValueError("Exchange rate needed before the first ECB rate (January 1999).")
    return pd.Series(s.to_numpy()[idx], index=pd.DatetimeIndex(dates))
