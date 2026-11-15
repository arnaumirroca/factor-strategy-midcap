"""Tests for the backtest: timing, returns, portfolio rules, costs, metrics and the hold-out guard."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics as mt
from src.backtest import run as rn
from src.backtest.engine import Config, simulate
from src.backtest.periods import fx_at, holding_windows, period_returns, trade_dates
from src.backtest.portfolio import sector_bounds, select


# --------------------------------------------------------------------------- #
# Timing and returns
# --------------------------------------------------------------------------- #
def test_trade_dates_kth_trading_day():
    cal = pd.bdate_range("2001-01-01", "2001-03-31")
    td = trade_dates(cal, 2)
    assert td[pd.Timestamp("2001-01-31")] == pd.Timestamp("2001-01-02")
    assert td[pd.Timestamp("2001-03-31")] == pd.Timestamp("2001-03-02")


def test_holding_window_runs_between_next_two_trade_dates():
    td = trade_dates(pd.bdate_range("2001-01-01", "2001-04-30"), 2)
    w = holding_windows([pd.Timestamp("2001-01-31")], td)
    assert w.loc["2001-01-31", "start"] == pd.Timestamp("2001-02-02")
    assert w.loc["2001-01-31", "end"] == pd.Timestamp("2001-03-02")


def _prices():
    days = pd.bdate_range("2001-02-01", "2001-03-05")
    adj = pd.DataFrame(index=days, dtype=float)
    adj["FLAT"] = 100.0
    adj["DIV"] = 100.0
    adj.loc["2001-02-15":, "DIV"] = 102.0       # 2% dividend reinvested in adjusted price
    adj["GONE"] = 100.0
    adj.loc["2001-02-20":, "GONE"] = np.nan     # stops trading at 90 on Feb 19
    adj.loc["2001-02-19", "GONE"] = 90.0
    adj["LATE"] = np.nan
    adj.loc["2001-02-10":, "LATE"] = 50.0       # not yet trading on the start date
    close = adj.copy()
    close["DIV"] = 100.0                         # price unchanged: the 2% is all dividend
    return adj, close


def test_period_returns_dividend_tax_delisting_and_tradability():
    adj, close = _prices()
    start, end = pd.Timestamp("2001-02-02"), pd.Timestamp("2001-03-02")
    r = period_returns(adj, close, start, end)
    assert r["FLAT"] == pytest.approx(0.0)
    assert r["DIV"] == pytest.approx(0.02 * 0.85)            # 15% withholding on the dividend
    assert r["GONE"] == pytest.approx(-0.10)                  # cashed out at last price
    assert "LATE" not in r.index                              # cannot be bought on the start date
    bankrupt = pd.Series({"GONE": pd.Timestamp("2001-02-19")})
    r2 = period_returns(adj, close, start, end, bankrupt)
    assert r2["GONE"] == pytest.approx(0.9 * 0.7 - 1)         # extra -30% for a delisting for cause


def test_fx_uses_latest_rate_on_or_before_date():
    fx = pd.Series([1.10, 1.20], index=pd.to_datetime(["2001-01-02", "2001-01-05"]))
    out = fx_at(fx, pd.to_datetime(["2001-01-04", "2001-01-05"]))
    assert list(out) == [1.10, 1.20]


# --------------------------------------------------------------------------- #
# Portfolio rules
# --------------------------------------------------------------------------- #
def _cands():
    sectors = ["Tech"] * 6 + ["Util"] * 4
    return pd.DataFrame({"ticker": [f"S{i}" for i in range(10)], "rank": range(1, 11), "sector": sectors,
                         "market_cap": 1.0})


def test_sector_bounds():
    mins, maxs = sector_bounds(pd.Series({"Tech": 0.6, "Util": 0.4}), 10, 0.05)
    assert mins == {"Tech": 6, "Util": 4} and maxs == {"Tech": 6, "Util": 4}


def test_select_respects_sector_minimum_and_maximum():
    c = _cands()
    chosen = select(c, set(), 4, 1.5, mins={"Util": 2}, maxs={"Tech": 2, "Util": 4})
    sec = c.set_index("ticker").loc[chosen, "sector"]
    assert (sec == "Util").sum() >= 2 and (sec == "Tech").sum() <= 2
    assert "S0" in chosen and "S6" in chosen               # best Tech and best Util


def test_buffer_keeps_holding_ranked_just_outside_top_n():
    c = _cands()
    chosen = select(c, {"S5"}, 4, 1.5, mins={}, maxs={"Tech": 10, "Util": 10})
    assert "S5" in chosen                                   # rank 6 <= 1.5 x 4
    chosen = select(c, {"S8"}, 4, 1.5, mins={}, maxs={"Tech": 10, "Util": 10})
    assert "S8" not in chosen                               # rank 9 is outside the buffer


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _sim_inputs(months=6, ret=0.0):
    F = pd.date_range("2001-01-31", periods=months, freq="ME")
    tick = [f"S{i}" for i in range(10)]
    scores = pd.DataFrame([(d, t, "Tech", float(10 - i), i + 1, 1.0) for d in F for i, t in enumerate(tick)],
                          columns=["date", "ticker", "sector", "composite", "rank", "market_cap"])
    for k in ("low_vol", "quality", "res_mom", "value"):
        scores[k] = scores["composite"] / 10
    windows = pd.DataFrame({"start": F + pd.Timedelta(days=2), "end": F + pd.Timedelta(days=30)}, index=F)
    periods = {d: pd.Series(ret, index=tick) for d in F}
    fx = pd.Series(1.0, index=F)
    return scores, windows, periods, fx


def test_contributions_do_not_count_as_returns():
    scores, windows, periods, fx = _sim_inputs()
    cfg = Config(fixed_n=4, commission_min_usd=0, commission_rate=0, spread_slippage_bps=0, fx_bps=0)
    res, _ = simulate(scores, windows, periods, fx, fx, cfg)
    assert res["ret_usd"].abs().max() < 1e-12               # flat prices, no costs -> 0% every month
    assert res["value_eur"].iloc[-1] == pytest.approx(10_000 + 5 * 1_500)


def test_costs_reduce_returns_and_match_the_model():
    scores, windows, periods, fx = _sim_inputs(months=1)
    res, trades = simulate(scores, windows, periods, fx, fx, Config(fixed_n=4))
    # First month: 4 buys of ~2,450 each -> max(1, 0.05%) + 7 bps per order, plus 1 bp on the 10,000 converted
    bps = Config().spread_slippage_bps / 1e4
    per_order = max(1.0, 0.0005 * 2450) + bps * 2450
    assert trades["cost_usd"].sum() == pytest.approx(4 * per_order, rel=1e-6)
    assert res["cost_eur"].iloc[0] == pytest.approx(4 * per_order + 1.0, rel=1e-6)
    assert res["ret_usd"].iloc[0] == pytest.approx(-res["cost_eur"].iloc[0] / 10_000)


def test_holding_count_switches_at_25000_eur():
    scores, windows, periods, fx = _sim_inputs(months=14)
    scores = pd.concat([scores.assign(ticker=scores["ticker"] + suffix) for suffix in ("", "a", "b", "c", "d")])
    scores["rank"] = scores.groupby("date")["composite"].rank(ascending=False, method="first")
    periods = {d: pd.Series(0.0, index=scores["ticker"].unique()) for d in windows.index}
    res, _ = simulate(scores, windows, periods, fx, fx, Config())
    assert res["n_target"].iloc[0] == 25 and res["n_target"].iloc[-1] == 40


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_basic_metrics():
    r = pd.Series([0.10, -0.50, 0.20, 0.0])
    assert mt.max_drawdown(r) == pytest.approx(-0.50)
    assert mt.cagr(pd.Series([0.01] * 12)) == pytest.approx(1.01 ** 12 - 1)


def test_deflated_sharpe_bar_rises_with_more_trials():
    rng = np.random.default_rng(0)
    ex = pd.Series(rng.normal(0.01, 0.04, 180))
    one, bar1 = mt.deflated_sharpe(ex, [0.2])
    many, bar2 = mt.deflated_sharpe(ex, list(rng.normal(0.15, 0.05, 50)))
    assert bar1 == 0.0 and bar2 > 0 and many < one


# --------------------------------------------------------------------------- #
# Hold-out guard and trial log
# --------------------------------------------------------------------------- #
def test_market_never_uses_holding_periods_after_2015():
    days = pd.bdate_range("2015-06-01", "2016-06-30")
    daily = pd.DataFrame([(d, "A", 10.0, 10.0) for d in days], columns=["date", "ticker", "close", "closeadj"])
    scores = pd.DataFrame({"date": pd.date_range("2015-06-30", "2016-03-31", freq="ME"), "ticker": "A"})
    spy = pd.DataFrame({"date": days, "close": 100.0, "closeadj": 100.0})
    fx = pd.Series(1.1, index=days)
    windows, *_ = rn.Market(scores, daily, pd.Series(dtype="datetime64[ns]"), spy, fx).setup(2)
    assert windows["end"].max() <= pd.Timestamp("2015-12-31")
    assert windows.index.max() == pd.Timestamp("2015-10-31")   # Nov 2015 -> Dec 2015 is the last period


def test_trial_log_records_each_setup_once(tmp_path, monkeypatch):
    monkeypatch.setattr(rn, "TRIAL_LOG", tmp_path / "trial_log.csv")
    a = rn.log_trial(Config(), 0.20, 180)
    b = rn.log_trial(Config(label="same settings, new name"), 0.20, 180)
    c = rn.log_trial(Config(fixed_n=30), 0.15, 180)
    assert len(a) == 1 and len(b) == 1 and len(c) == 2


# --------------------------------------------------------------------------- #
# Trial 2: waiting period and trimming only above the single-stock limit
# --------------------------------------------------------------------------- #
def test_defaults_are_the_agreed_dmf_us_02_rules():
    c = Config()
    assert (c.trim_rule, c.exit_after_months, c.spread_slippage_bps) == ("limit", 2, 20.0)
    assert Config(label="other name").key() == c.key()
    assert rn.CURRENT.key() == c.key()


def test_select_keeps_must_keep_names():
    c = _cands()
    chosen = select(c, {"S9"}, 4, 1.5, mins={}, maxs={"Tech": 10, "Util": 10}, must_keep={"S9"})
    assert "S9" in chosen and len(chosen) == 4


# With only 4 holdings every position is far above the single-stock limit, so the waiting-period
# tests use the "band" sizing rule to keep trims out of the way.
def _drop_inputs(months):
    """S0 ranks first, then falls to last for the remaining months."""
    scores, windows, periods, fx = _sim_inputs(months=months)
    later = scores["date"] > scores["date"].min()
    scores.loc[later & (scores["ticker"] == "S0"), "rank"] = 99
    scores.loc[later & (scores["ticker"] == "S0"), "composite"] = -1.0
    return scores, windows, periods, fx


def test_waiting_period_delays_the_sale_by_one_month():
    scores, windows, periods, fx = _drop_inputs(3)
    _, t1 = simulate(scores, windows, periods, fx, fx, Config(fixed_n=4, exit_after_months=1, trim_rule="band"))
    _, t2 = simulate(scores, windows, periods, fx, fx, Config(fixed_n=4, exit_after_months=2, trim_rule="band"))
    sell = lambda t: t[(t["ticker"] == "S0") & (t["usd"] < 0)]["formation"]
    F = sorted(windows.index)
    assert list(sell(t1)) == [F[1]]                          # sold at the first month-end outside
    assert list(sell(t2)) == [F[2]]                          # sold at the second


def test_waiting_stock_that_recovers_is_kept():
    scores, windows, periods, fx = _sim_inputs(months=3)
    m2 = scores["date"] == sorted(windows.index)[1]
    scores.loc[m2 & (scores["ticker"] == "S0"), "rank"] = 99
    res, t = simulate(scores, windows, periods, fx, fx, Config(fixed_n=4, exit_after_months=2, trim_rule="band"))
    assert not ((t["ticker"] == "S0") & (t["usd"] < 0)).any()
    assert res.attrs["waiting_stats"]["recovered"] == 1


def test_limit_rule_trims_only_above_limit_plus_drift():
    scores, windows, periods, fx = _sim_inputs(months=2)
    F = sorted(windows.index)
    # S0 grows 40% in month 1: about 34% of 4 equal holdings -> far above 5% + 1pp -> trimmed to 5%
    periods[F[0]] = pd.Series({t: (0.40 if t == "S0" else 0.0) for t in periods[F[0]].index})
    cfg = Config(fixed_n=4, trim_rule="limit", exit_after_months=2)
    _, t = simulate(scores, windows, periods, fx, fx, cfg)
    s0 = t[(t["ticker"] == "S0") & (t["formation"] == F[1])]
    assert len(s0) == 1 and s0["usd"].iloc[0] < 0
    # S1 drifted down to ~0.86x target: inside the band, so no trade under the limit rule
    small = Config(fixed_n=4, trim_rule="limit", exit_after_months=2, max_weight_small=0.5)
    _, t = simulate(scores, windows, periods, fx, fx, small)
    assert not ((t["formation"] == F[1]) & (t["ticker"] == "S0") & (t["usd"] < 0)).any()
