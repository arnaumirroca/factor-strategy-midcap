"""Raw characteristics behind the low-volatility, quality and value scores (Design Doc 6.1).

Each function returns a raw number where HIGHER = BETTER for the strategy, so every
score can be ranked the same way afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VOL_DAYS = 252
MIN_VOL_DAYS = 200


# --------------------------------------------------------------------------- #
# Low volatility (6.1.4)
# --------------------------------------------------------------------------- #
def daily_volatility(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Rolling 252-day standard deviation of daily returns (wide: days x tickers)."""
    return daily_returns.rolling(VOL_DAYS, min_periods=MIN_VOL_DAYS).std()


def at_month_ends(daily_wide: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Value on the last trading day of each month, indexed by calendar month-end."""
    last_day = daily_wide.groupby(daily_wide.index + pd.offsets.MonthEnd(0)).tail(1)
    last_day.index = (last_day.index + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    return last_day.reindex(month_ends)


def low_vol_raw(vol: pd.Series) -> pd.Series:
    """Lower volatility is better, so flip the sign."""
    return -vol


# --------------------------------------------------------------------------- #
# Quality (6.1.3)
# --------------------------------------------------------------------------- #
def operating_profitability(revenue, cor, sgna, intexp, equity) -> pd.Series:
    """(Revenue - cost of revenue - SG&A - interest) / book equity, trailing 12 months.

    Missing cost lines count as zero when at least one is reported (Fama-French
    convention). Negative or zero equity gets the lowest possible value.
    """
    costs = pd.concat([cor, sgna, intexp], axis=1)
    has_cost = costs.notna().any(axis=1)
    op = (revenue - costs.fillna(0.0).sum(axis=1)) / equity
    op = op.where(has_cost & revenue.notna())
    return op.mask(equity <= 0, -np.inf)


def low_leverage(debt, assets) -> pd.Series:
    """Minus debt / total assets: less debt is better. Missing debt is treated as none."""
    return -(debt.fillna(0.0) / assets.where(assets > 0))


# --------------------------------------------------------------------------- #
# Value (6.1.2)
# --------------------------------------------------------------------------- #
def value_ratios(equity, earnings_ttm, fcf_ttm, market_cap) -> pd.DataFrame:
    """Book-to-market, earnings yield and free-cash-flow yield.

    Negative earnings or cash flow give negative yields, which rank lowest naturally.
    """
    mc = market_cap.where(market_cap > 0)
    return pd.DataFrame({"bm": equity / mc, "ep": earnings_ttm / mc, "fcfp": fcf_ttm / mc})
