"""Tests for the mid/small-cap universe: market values, split adjustment and rank limits."""

import numpy as np
import pandas as pd
import pytest

from src.data import size_universe as su


def _inputs():
    """Sharadar restates share counts for later splits; prices in `monthly` are as traded."""
    d = pd.Timestamp("2005-06-30")
    monthly = pd.DataFrame({
        "date": [d] * 5,
        "ticker": ["BIG", "MID", "SPLIT", "CHEAP", "OLD"],
        "price": [100.0, 50.0, 20.0, 3.0, 40.0],
        "last_date": [d - pd.Timedelta(days=1)] * 4 + [d - pd.Timedelta(days=20)],
    })
    # SPLIT does a 2-for-1 split years later: k = 2 now, and its share count is restated to 2e8
    k = pd.DataFrame({"date": [d] * 5, "ticker": ["BIG", "MID", "SPLIT", "CHEAP", "OLD"],
                      "k": [1.0, 1.0, 2.0, 1.0, 1.0]})
    fund = pd.DataFrame({"ticker": ["BIG", "MID", "SPLIT", "CHEAP", "OLD"], "dimension": "ARQ",
                         "filed": pd.to_datetime(["2005-05-01"] * 5),
                         "sharesbas": [1e9, 1e8, 2e8, 1e9, 1e8],
                         "marketcap": [np.nan] * 5, "price": [np.nan] * 5})
    return monthly, k, fund


def test_market_value_uses_one_split_basis_and_filters():
    monthly, k, fund = _inputs()
    r = su.size_ranks(monthly, k, fund, {"BIG", "MID", "SPLIT", "CHEAP", "OLD"}).set_index("ticker")
    assert "CHEAP" not in r.index                          # price below $5
    assert "OLD" not in r.index                            # stopped trading mid-month
    assert r.loc["SPLIT", "market_cap"] == pytest.approx(20.0 * 1e8)   # 1e8 real shares at $20
    assert list(r.sort_values("size_rank").index) == ["BIG", "MID", "SPLIT"]


def test_shares_come_from_sharadars_market_value_when_available():
    monthly, k, fund = _inputs()
    fund.loc[fund["ticker"] == "MID", ["marketcap", "price"]] = [3e9, 10.0]   # 3e8 shares incl. share factor
    r = su.size_ranks(monthly, k, fund, {"BIG", "MID", "SPLIT"}).set_index("ticker")
    assert r.loc["MID", "market_cap"] == pytest.approx(50.0 * 3e8)


def test_filing_on_the_month_end_is_not_used_and_stale_filings_are_ignored():
    monthly, k, fund = _inputs()
    fund.loc[fund["ticker"] == "MID", "filed"] = pd.Timestamp("2005-06-30")      # same day: not yet public
    fund.loc[fund["ticker"] == "BIG", "filed"] = pd.Timestamp("2004-01-01")      # over 400 days old
    r = su.size_ranks(monthly, k, fund, {"BIG", "MID", "SPLIT"})
    assert set(r["ticker"]) == {"SPLIT"}


def test_members_keep_only_the_chosen_ranks(monkeypatch):
    monkeypatch.setattr(su, "RANK_FROM", 2)
    monkeypatch.setattr(su, "RANK_TO", 3)
    ranks = pd.DataFrame({"date": pd.Timestamp("2005-06-30"), "ticker": list("ABCD"), "market_cap": [4, 3, 2, 1],
                          "size_rank": [1, 2, 3, 4]})
    assert list(su.members(ranks)["ticker"]) == ["B", "C"]


def test_month_end_split_factor_takes_the_last_day():
    daily = pd.DataFrame({"ticker": ["A", "A", "A"], "date": pd.to_datetime(["2005-06-28", "2005-06-30", "2005-07-29"]),
                          "close": [10.0, 10.0, 10.0], "closeunadj": [20.0, 10.0, 10.0]})
    k = su.month_end_split_factor(daily)
    assert list(k["k"]) == [1.0, 1.0]
    assert list(k["date"]) == [pd.Timestamp("2005-06-30"), pd.Timestamp("2005-07-31")]
