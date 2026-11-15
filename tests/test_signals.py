"""Tests for the four scores, the composite and the hold-out rule (Stage 04)."""

import numpy as np
import pandas as pd
import pytest

from src.signals import build as bd
from src.signals import characteristics as ch
from src.signals.common import combine, rank_score
from src.signals.residual_momentum import FACTORS, residual_momentum_at


# --------------------------------------------------------------------------- #
# Ranking and combining
# --------------------------------------------------------------------------- #
def test_rank_score_scales_to_minus_one_plus_one():
    s = rank_score(pd.Series([10.0, 30.0, 20.0, np.nan, -np.inf]))
    assert s.iloc[[0, 1, 2, 4]].round(3).tolist() == [-0.333, 1.0, 0.333, -1.0]
    assert np.isnan(s.iloc[3])                             # missing stays missing


def test_rank_score_within_groups():
    vals = pd.Series([1.0, 2.0, 100.0, 200.0])
    sec = pd.Series(["Tech", "Tech", "Util", "Util"])
    s = rank_score(vals, sec)
    assert list(s) == [-1.0, 1.0, -1.0, 1.0]            # each sector ranked on its own
    assert rank_score(pd.Series([5.0]), pd.Series(["Solo"])).iloc[0] == 0.0


def test_combine_weights_and_missing_scores():
    df = pd.DataFrame({"low_vol": [1.0, 1.0, np.nan], "quality": [1.0, 1.0, np.nan],
                       "res_mom": [1.0, np.nan, np.nan], "value": [1.0, -1.0, 1.0]})
    c = combine(df)
    assert c.iloc[0] == pytest.approx(1.0)
    assert c.iloc[1] == pytest.approx(0.30 + 0.30 + 0 - 0.15)   # missing momentum counts as 0
    assert np.isnan(c.iloc[2])                                   # only 1 of 4 scores


# --------------------------------------------------------------------------- #
# Residual momentum
# --------------------------------------------------------------------------- #
def _factor_data(n=48, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    fac = pd.DataFrame(rng.normal(0, 0.04, (n, 3)), index=idx, columns=FACTORS)
    return idx, fac, rng


def test_residual_momentum_rewards_stock_specific_gains_only():
    idx, fac, rng = _factor_data()
    F = idx[40]
    noise = lambda: rng.normal(0, 0.005, len(idx))
    up = 1.2 * fac["Mkt-RF"] + noise()
    up.iloc[29:40] += 0.03                               # stock-specific gains in F-11 .. F-1
    down = 0.8 * fac["Mkt-RF"] + noise()
    down.iloc[29:40] -= 0.03
    beta_only = 1.5 * fac["Mkt-RF"] + noise()            # rises with the market, nothing specific
    ex = pd.DataFrame({"UP": up, "DOWN": down, "BETA": beta_only})
    s = residual_momentum_at(ex, fac, F, ["UP", "DOWN", "BETA"])
    assert s["UP"] > 2 and s["DOWN"] < -2 and abs(s["BETA"]) < s["UP"]


def test_residual_momentum_ignores_the_formation_month():
    idx, fac, rng = _factor_data()
    F = idx[40]
    ex = pd.DataFrame({"A": fac["Mkt-RF"] + rng.normal(0, 0.01, len(idx))})
    before = residual_momentum_at(ex, fac, F, ["A"])["A"]
    ex.loc[F, "A"] = 0.80                                # huge return in month F itself
    assert residual_momentum_at(ex, fac, F, ["A"])["A"] == pytest.approx(before)


def test_residual_momentum_handles_gaps_and_short_history():
    idx, fac, rng = _factor_data()
    F = idx[40]
    ex = pd.DataFrame({"GAPS": fac["Mkt-RF"] + rng.normal(0, 0.01, len(idx)),
                       "SHORT": fac["Mkt-RF"] + rng.normal(0, 0.01, len(idx))})
    ex.iloc[6:9, 0] = np.nan                              # 3 missing months: still 33 of 36
    ex.iloc[:20, 1] = np.nan                              # only 20 months: too short
    s = residual_momentum_at(ex, fac, F, ["GAPS", "SHORT"])
    assert np.isfinite(s["GAPS"]) and np.isnan(s["SHORT"])


# --------------------------------------------------------------------------- #
# Characteristics
# --------------------------------------------------------------------------- #
def test_operating_profitability_and_leverage():
    rev = pd.Series([100.0, 100.0, 100.0, 100.0])
    cor = pd.Series([50.0, np.nan, 50.0, np.nan])
    sgna = pd.Series([20.0, 20.0, 20.0, np.nan])
    intexp = pd.Series([10.0, np.nan, 10.0, np.nan])
    eq = pd.Series([100.0, 100.0, -5.0, 100.0])
    op = ch.operating_profitability(rev, cor, sgna, intexp, eq)
    assert op.iloc[0] == pytest.approx(0.20)
    assert op.iloc[1] == pytest.approx(0.80)              # missing cost lines count as zero
    assert op.iloc[2] == -np.inf                          # negative equity ranks last
    assert np.isnan(op.iloc[3])                           # no cost lines at all
    lev = ch.low_leverage(pd.Series([50.0, np.nan]), pd.Series([100.0, 100.0]))
    assert list(lev) == [-0.5, -0.0]


def test_month_end_sampling_uses_last_trading_day():
    days = pd.to_datetime(["2001-01-30", "2001-01-31", "2001-02-27", "2001-02-28"])
    wide = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=days)
    out = ch.at_month_ends(wide, pd.DatetimeIndex(["2001-01-31", "2001-02-28"]))
    assert list(out["A"]) == [2.0, 4.0]


def test_score_month_is_sector_relative_except_low_vol():
    g = pd.DataFrame({
        "sector": ["Tech", "Tech", "Util", "Util"],
        "low_vol_raw": [-0.04, -0.03, -0.01, -0.02],
        "op": [0.30, 0.10, 0.08, 0.05], "leverage_raw": [-0.1, -0.1, -0.5, -0.5],
        "bm": [0.2, 0.1, 0.9, 0.5], "ep": [0.05, 0.02, 0.07, 0.06], "fcfp": [0.04, 0.01, 0.05, 0.03],
        "res_mom_raw": [1.0, -1.0, 0.5, 2.0]})
    out = bd.score_month(g)
    assert list(out["quality"]) == [1.0, -1.0, 1.0, -1.0]  # best in each sector, despite Tech's higher OP
    assert out["low_vol"].iloc[2] == 1.0                     # least volatile overall
    assert out["rank"].min() == 1


# --------------------------------------------------------------------------- #
# Hold-out rule
# --------------------------------------------------------------------------- #
def test_outcomes_never_come_from_the_hold_out():
    scores = pd.DataFrame({"date": pd.to_datetime(["2015-11-30", "2015-12-31", "2020-01-31"]),
                           "ticker": "A", "composite": 0.5})
    monthly = pd.DataFrame({"date": pd.to_datetime(["2015-12-31", "2016-01-31", "2020-02-29"]),
                            "ticker": "A", "ret": [0.01, 0.02, 0.03]})
    df = bd.in_sample_outcomes(scores, monthly)
    assert list(df["date"]) == [pd.Timestamp("2015-11-30")]  # Dec-2015 return only
    assert df["next_ret"].iloc[0] == 0.01


# --------------------------------------------------------------------------- #
# Panel and scores, end to end on a small synthetic market
# --------------------------------------------------------------------------- #
def synthetic_market(n=60, months=60, seed=3):
    rng = np.random.default_rng(seed)
    me = pd.date_range("1998-01-31", periods=months, freq="ME")
    tick = [f"S{i:02d}" for i in range(n)]
    ff3 = pd.DataFrame(rng.normal(0, 0.04, (months, 3)), index=me, columns=FACTORS)
    ff3["RF"] = 0.002
    vol = rng.uniform(0.01, 0.03, n)
    days = pd.bdate_range(me[0] - pd.offsets.MonthBegin(1), me[-1])
    dr = rng.normal(0.0003, 1, (len(days), n)) * vol
    px = 30 * np.exp(np.cumsum(dr, axis=0))
    px[:, 0] = 3.0                                          # S00 trades below $5: excluded
    daily = pd.DataFrame({"date": np.repeat(days, n), "ticker": np.tile(tick, len(days)),
                          "close": px.ravel(), "closeadj": px.ravel(), "closeunadj": px.ravel()})
    wide = pd.DataFrame(px, index=days, columns=tick)
    m_px = ch.at_month_ends(wide, me)
    monthly = m_px.pct_change().stack().rename("ret").reset_index()
    monthly.columns = ["date", "ticker", "ret"]
    monthly = monthly.merge(m_px.stack().rename("price").reset_index().set_axis(["date", "ticker", "price"], axis=1))
    monthly["adv_usd"] = 50e6
    membership = pd.DataFrame([(d, t) for d in me for t in tick], columns=["date", "ticker"])
    sectors = ["Technology", "Utilities", "Healthcare"]
    tickers = pd.DataFrame({"ticker": tick, "name": tick, "sector": [sectors[i % 3] for i in range(n)]})
    q = pd.date_range("1997-03-31", me[-1], freq="QE")
    rows = []
    for i, t in enumerate(tick):
        for rp in q:
            for dim in ("ARQ", "ART"):
                rows.append({"ticker": t, "dimension": dim, "reportperiod": rp, "filed": rp + pd.Timedelta(days=40),
                             "equity": 100.0 + i, "assets": 300.0, "debt": float(i), "sharesbas": 1e6,
                             "revenue": 200.0 + 3 * i, "cor": 100.0, "sgna": 40.0, "intexp": 5.0,
                             "netinccmn": 20.0 + i % 7, "fcf": 15.0 - i % 5})
    fund = pd.DataFrame(rows)
    return membership, monthly, tickers, fund, daily, ff3


def test_build_panel_and_scores_end_to_end():
    membership, monthly, tickers, fund, daily, ff3 = synthetic_market()
    panel = bd.build_panel(membership, monthly, tickers, fund, daily, ff3)
    assert not panel.loc[panel["ticker"] == "S00", "in_universe"].any()      # price below $5
    early = panel["date"] < pd.Timestamp("2000-06-30")
    assert not panel.loc[early, "in_universe"].any()                         # under 30 months of history
    scores = bd.build_scores(panel)
    last = scores[scores["date"] == scores["date"].max()]
    assert len(last) == 59
    assert last["composite"].notna().all()
    assert last[["low_vol", "quality", "res_mom", "value"]].abs().max().max() <= 1.0
    # Lowest-volatility stocks must get the best low-vol scores
    vol_rank = last.set_index("ticker")["vol"].rank()
    lv = last.set_index("ticker")["low_vol"]
    assert vol_rank.corr(-lv) > 0.99
