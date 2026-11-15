"""Tests for the Gate 1 momentum rebuild, on small hand-made data."""

import numpy as np
import pandas as pd
import pytest

from src.data import gate1_momentum as g1


def _months(n, start="2000-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_momentum_skips_the_most_recent_month():
    idx = _months(14)
    r = pd.DataFrame({"A": 0.0}, index=idx)
    r.iloc[12, 0] = 0.50          # month F (row 12): must NOT count for formation at F
    r.iloc[11, 0] = 0.10          # month F-1: must count
    r.iloc[0, 0] = 0.99           # month F-12: outside the window
    mom = g1.momentum_signal(r)
    assert mom.iloc[12, 0] == pytest.approx(0.10)


def test_momentum_needs_enough_history():
    r = pd.DataFrame({"A": 0.01}, index=_months(14))
    r.iloc[3:9, 0] = np.nan       # only 5 of the 11 months available at F = row 12
    assert np.isnan(g1.momentum_signal(r).iloc[12, 0])


def test_shares_scaled_for_split_after_filing():
    # 2-for-1 split after the filing: unadjusted/adjusted price ratio halves, shares double.
    out = g1.split_adjusted_shares(pd.Series([100.0]), pd.Series([2.0]), pd.Series([1.0]))
    assert out.iloc[0] == 200.0


def test_umd_one_month_value_weighted_winners_minus_losers():
    n = 40
    tick = [f"S{i}" for i in range(n)]
    size = pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, 10.0), index=tick)   # half small, half big
    mom = pd.Series(np.linspace(-0.5, 0.5, n), index=tick)
    next_ret = pd.Series(np.where(mom >= mom.quantile(0.7), 0.10, np.where(mom <= mom.quantile(0.3), -0.10, 0.0)),
                         index=tick)
    is_nyse = pd.Series(True, index=tick)
    res = g1.umd_one_month(size, mom, next_ret, is_nyse)
    assert res["umd"] == pytest.approx(0.20)
    assert res["n"] == n


def test_umd_uses_nyse_breakpoints_only():
    tick = [f"S{i}" for i in range(30)]
    size = pd.Series(1.0, index=tick)
    mom = pd.Series(np.arange(30, dtype=float), index=tick)
    next_ret = pd.Series(0.0, index=tick)
    too_few_nyse = pd.Series([i < 10 for i in range(30)], index=tick)
    assert np.isnan(g1.umd_one_month(size, mom, next_ret, too_few_nyse)["umd"])


def test_compare_reports_correlation():
    idx = _months(60)
    x = pd.Series(np.random.default_rng(0).normal(0, 0.04, 60), index=idx)
    stats = g1.compare(x, x * 1.1)
    assert stats["correlation"] == pytest.approx(1.0)
    assert stats["beta"] == pytest.approx(1 / 1.1, rel=1e-6)
    with pytest.raises(SystemExit):
        g1.compare(x.iloc[:10], x.iloc[:10])
