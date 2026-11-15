"""In-sample backtest, Aug 2000 - Dec 2015 (Stage 05, Design Doc Section 10).

Run from the project folder, after the signals are built and the benchmark downloaded:

    .\\.venv\\Scripts\\python.exe -m src.data.benchmark_fx      (once)
    .\\.venv\\Scripts\\python.exe -m src.backtest.run

Output (reports/)
-----------------
backtest_insample.txt           Results vs the S&P 500 in EUR, the Section 10.3 and 10.4
                                checks, calendar years, factor exposures, deflated Sharpe
backtest_insample.png           Growth of EUR 1, falls from peak, rolling 3-year excess return
backtest_insample_monthly.csv   Month by month: returns, value, costs, holdings
backtest_insample_trades.csv    Every simulated order
trial_log.csv                   Every distinct setup ever run (Section 10.6)

The hold-out (2016 onwards) is never touched: every holding period must end by
31 December 2015, and the run stops if one would not.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

from src.backtest import metrics as mt
from src.backtest.engine import Config, simulate
from src.backtest.periods import fx_at, holding_windows, period_returns, trade_dates
from src.data import sharadar as sh
from src.data.benchmark_fx import FX_PATH, SPY_PATH
from src.signals.build import OUT_PATH as SCORES_PATH
from src.signals.common import WEIGHTS

IN_SAMPLE_END = pd.Timestamp("2015-12-31")
REPORTS = sh.PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS / "backtest_insample.txt"
CHART_PATH = REPORTS / "backtest_insample.png"
MONTHLY_PATH = REPORTS / "backtest_insample_monthly.csv"
TRADES_PATH = REPORTS / "backtest_insample_trades.csv"
TRIAL_LOG = REPORTS / "trial_log.csv"
# DMF-US-02 (decided 14/11/2026, before running): the S&P 500 project's trial-2 rules applied to
# US stocks ranked 501-1,500 by market value, with trading costs of 20 bps instead of 7.
# The trial log starts with the S&P 500 project's two trials, so the deflated Sharpe counts them.
CURRENT = Config(label="Trial 4: mid/small caps (ranks 501-1,500)")

SUB_PERIODS = [("Aug 2000-2003", "2000", "2003"), ("2004-2009", "2004", "2009"), ("2010-2015", "2010", "2015")]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load():
    for p, how in [(SCORES_PATH, "python -m src.signals.build"), (SPY_PATH, "python -m src.data.benchmark_fx"),
                   (FX_PATH, "python -m src.data.benchmark_fx")]:
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run: {how}")
    scores = sh.read_parquet(SCORES_PATH)
    scores["date"] = scores["date"].astype("datetime64[ns]")
    names = set(scores["ticker"])
    daily = sh.read_parquet(sh.CLEAN_DIR / "prices_daily.parquet", columns=["ticker", "date", "close", "closeadj"],
                            read_dictionary=["ticker"])
    daily = daily[daily["ticker"].isin(names)].copy()
    daily["ticker"] = daily["ticker"].astype(str)
    daily["date"] = daily["date"].astype("datetime64[ns]")
    monthly = sh.read_parquet(sh.CLEAN_DIR / "monthly.parquet", columns=["ticker", "delisting_adjusted"])
    for_cause = set(monthly.loc[monthly["delisting_adjusted"], "ticker"]) & names
    last_trade = daily.groupby("ticker")["date"].max()
    for_cause_last = last_trade[last_trade.index.isin(for_cause)]
    spy = sh.read_parquet(SPY_PATH)
    fx = sh.read_parquet(FX_PATH).set_index("date")["usd_per_eur"]
    ff_dir = sh.PROJECT_ROOT / "data" / "clean" / "fama_french"
    ff = {}
    for k in ("us_ff3", "us_ff5", "us_mom"):
        f = sh.read_parquet(ff_dir / f"{k}.parquet")
        f = f.set_index("date") if "date" in f.columns else f
        f.index = pd.to_datetime(f.index).astype("datetime64[ns]")
        ff[k] = f
    return scores, daily, for_cause_last, spy, fx, ff


class Market:
    """Prices and trade calendars, with holding-period returns cached per trade day."""

    def __init__(self, scores, daily, for_cause_last, spy, fx):
        self.scores = scores
        self.adj = daily.pivot(index="date", columns="ticker", values="closeadj")
        self.close = daily.pivot(index="date", columns="ticker", values="close")
        self.spy = spy.set_index("date")
        self.calendar = pd.DatetimeIndex(self.spy.index)
        self.for_cause_last = for_cause_last
        self.fx = fx
        self._cache = {}

    def setup(self, trade_day: int):
        if trade_day in self._cache:
            return self._cache[trade_day]
        tdates = trade_dates(self.calendar, trade_day)
        windows = holding_windows(sorted(self.scores["date"].unique()), tdates)
        windows = windows[windows["end"] <= IN_SAMPLE_END]
        if (windows["end"] > IN_SAMPLE_END).any():
            raise SystemExit("Refusing to use hold-out data.")
        periods, bench = {}, {}
        for F, w in windows.iterrows():
            periods[F] = period_returns(self.adj, self.close, w["start"], w["end"], self.for_cause_last)
            b = period_returns(self.spy[["closeadj"]].rename(columns={"closeadj": "SPY"}),
                               self.spy[["close"]].rename(columns={"close": "SPY"}), w["start"], w["end"])
            bench[F] = float(b["SPY"])
        fx0 = pd.Series(fx_at(self.fx, windows["start"]).to_numpy(), index=windows.index)
        fx1 = pd.Series(fx_at(self.fx, windows["end"]).to_numpy(), index=windows.index)
        self._cache[trade_day] = (windows, periods, bench, fx0, fx1)
        return self._cache[trade_day]

    def run(self, cfg: Config):
        windows, periods, bench, fx0, fx1 = self.setup(cfg.trade_day)
        res, trades = simulate(self.scores, windows, periods, fx0, fx1, cfg)
        b_usd = pd.Series({(F + pd.offsets.MonthEnd(1)).normalize(): bench[F] for F in windows.index})
        res["bench_usd"] = b_usd.reindex(res.index)
        res["bench_eur"] = (1 + res["bench_usd"]) * fx0.reindex(res["formation"]).to_numpy() \
            / fx1.reindex(res["formation"]).to_numpy() - 1
        return res, trades


# --------------------------------------------------------------------------- #
# Trial log (Section 10.6)
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=sh.PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def log_trial(cfg: Config, monthly_sharpe: float, months: int) -> list[float]:
    """Add the setup to the log if new; return the monthly Sharpes of all logged trials."""
    rows = []
    if TRIAL_LOG.exists():
        with open(TRIAL_LOG, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    TRIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if cfg.key() not in {r["config_key"] for r in rows}:
        rows.append({"trial": len(rows) + 1, "date": f"{date.today():%Y-%m-%d}", "config_key": cfg.key(),
                     "description": cfg.label, "monthly_sharpe": f"{monthly_sharpe:.5f}", "months": months,
                     "commit": git_commit(), "settings": repr(cfg)})
        with open(TRIAL_LOG, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return [float(r["monthly_sharpe"]) for r in rows]


# --------------------------------------------------------------------------- #
# Report helpers
# --------------------------------------------------------------------------- #
def pct(x, sign=True):
    return "   n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:+.1%}" if sign else f"{x:.1%}")


def turnover(res: pd.DataFrame) -> float:
    years = len(res) / 12
    return float(res["sells_eur"].sum() / res["value_eur"].mean() / years)


def save_chart(res: pd.DataFrame, s: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  (chart not saved: {exc})")
        return
    r, b = res["ret_eur"], res["bench_eur"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    ax[0].plot((1 + r).cumprod(), label="Strategy (net of costs)")
    ax[0].plot((1 + b).cumprod(), label="S&P 500 (SPY, net of withholding)")
    ax[0].set_yscale("log")
    ax[0].set_title(f"Growth of EUR 1, in-sample {res.index.min():%b %Y} - {res.index.max():%b %Y}")
    ax[0].legend(loc="upper left")
    ax[1].fill_between(r.index, mt.drawdowns(r), 0, alpha=0.4, label="Strategy")
    ax[1].plot(mt.drawdowns(b), color="C1", lw=1, label="S&P 500")
    ax[1].set_title("Fall from previous peak")
    ax[1].legend(loc="lower left")
    roll = (1 + r).rolling(36).apply(np.prod, raw=True) ** (1 / 3) - (1 + b).rolling(36).apply(np.prod, raw=True) ** (1 / 3)
    ax[2].axhline(0, color="grey", lw=0.8)
    ax[2].plot(roll, color="C2")
    ax[2].set_title("Rolling 3-year excess return a year (strategy minus S&P 500)")
    for a in ax:
        a.grid(alpha=0.3)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    try:
        fig.savefig(CHART_PATH, dpi=150)
    except OSError:
        # Windows refuses to overwrite a picture that is open in a viewer (e.g. a VS Code tab).
        alt = CHART_PATH.with_name(CHART_PATH.stem + "_new.png")
        try:
            fig.savefig(alt, dpi=150)
            print(f"  ({CHART_PATH.name} is open in another program; chart saved as {alt.name})")
        except OSError as exc:
            print(f"  (chart not saved: {exc})")
    plt.close(fig)


def robustness_configs(base: Config) -> list[Config]:
    """Section 10.4 variants, each changing one thing from the current setup."""
    v = lambda **kw: dataclasses.replace(base, **kw)
    out = [v(label=f"{n} holdings throughout", fixed_n=n) for n in (25, 40, 50)]
    out.append(v(label="2x trading costs", cost_multiplier=2.0))
    out.append(v(label="Trade 5 trading days later", trade_day=7))
    out.append(v(label="Equal weights (25% each)", weights=tuple((k, 0.25) for k in WEIGHTS)))
    for drop in WEIGHTS:
        w = tuple((k, x) for k, x in WEIGHTS.items() if k != drop)
        out.append(v(label=f"Without {drop}", weights=w, min_signals=2))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run() -> int:
    print("Stage 05: in-sample backtest (holding periods ending by Dec 2015)")
    scores, daily, for_cause_last, spy, fx, ff = load()
    mkt = Market(scores, daily, for_cause_last, spy, fx)
    del daily

    base = CURRENT
    print(f"  {base.label}...", flush=True)
    res, trades = mkt.run(base)
    if res.empty:
        raise SystemExit("No months simulated. Check that scores exist before 2016.")
    rf = ff["us_ff3"]["RF"].reindex(res.index).fillna(0.0)
    r, b = res["ret_eur"], res["bench_eur"]
    s = mt.summary(r, b, rf)
    excess_m = res["ret_usd"] - rf
    sr_m = float(excess_m.mean() / excess_m.std(ddof=1))
    trial_srs = log_trial(base, sr_m, len(res))
    dsr, bar = mt.deflated_sharpe(excess_m, trial_srs)
    years = len(res) / 12
    to = turnover(res)
    cost_pa = float(res["cost_eur"].sum() / res["value_eur"].mean() / years)

    ffx = ff["us_ff5"][["Mkt-RF", "SMB", "HML", "RMW", "CMA"]].join(ff["us_mom"]["MOM"])
    reg = mt.ols(excess_m, ffx.reindex(res.index))

    REPORTS.mkdir(parents=True, exist_ok=True)
    res.to_csv(MONTHLY_PATH, float_format="%.6f")
    trades.to_csv(TRADES_PATH, index=False, float_format="%.2f")
    save_chart(res, s)

    L = [f"In-sample backtest - {date.today():%d %b %Y} - {base.label}",
         f"Holding months {res.index.min():%b %Y} to {res.index.max():%b %Y} ({len(res)} months). "
         f"Hold-out (2016 onwards) not used.",
         f"Setup: EUR {base.start_capital_eur:,.0f} start + EUR {base.contribution_eur:,.0f}/month; "
         f"{base.n_small} holdings below EUR {base.n_switch_eur:,.0f}, then {base.n_large}; buffer {base.buffer}x; "
         f"sector +/-{base.sector_band:.0%}; trade on trading day {base.trade_day}; trading costs "
         f"{base.spread_slippage_bps:.0f} bps + commission per Section 9.2.",
         "Universe: US common stocks ranked 501-1,500 by market value (price above $5, volume above $2m a day). "
         "Benchmark: S&P 500.",
         (f"Selling: after {base.exit_after_months} consecutive month-ends outside the top {base.buffer} x N; "
          f"winners trimmed only above the single-stock limit + {base.drift_pp:.0%}."
          if base.trim_rule == "limit" else
          f"Selling: as soon as a stock ranks outside the top {base.buffer} x N; positions resized to target "
          f"outside {base.resize_band[0]:.0%}-{base.resize_band[1]:.0%} of target weight."),
         "Returns are time-weighted (contributions do not inflate them), in EUR, after all costs and 15% dividend withholding.",
         "", f"{'':<34}{'Strategy':>10}{'S&P 500':>10}{'Difference':>12}",
         f"{'Return a year':<34}{pct(s['cagr']):>10}{pct(s['bench_cagr']):>10}{pct(s['excess']):>12}",
         f"{'Volatility a year':<34}{pct(s['vol'], False):>10}{pct(s['bench_vol'], False):>10}{pct(s['vol'] - s['bench_vol']):>12}",
         f"{'Sharpe ratio':<34}{s['sharpe']:>10.2f}{s['bench_sharpe']:>10.2f}{s['sharpe'] - s['bench_sharpe']:>+12.2f}",
         f"{'Worst fall from peak':<34}{pct(s['max_dd']):>10}{pct(s['bench_max_dd']):>10}{pct(s['max_dd'] - s['bench_max_dd']):>12}",
         "", f"Tracking error {pct(s['te'], False)}  |  Information ratio {s['ir']:.2f}  |  Beta {s['beta']:.2f}  |  "
             f"Months beating the S&P 500 {s['hit']:.0%}  |  Worst relative fall {pct(s['rel_dd'])}",
         f"Turnover {to:.0%} a year (value of stocks sold in a year / average portfolio value)  |  Costs {cost_pa:.2%} of capital a year  |  Orders {res['orders'].sum() / years:.0f} a year  |  "
         f"Final value EUR {res['value_eur'].iloc[-1]:,.0f} (contributed EUR {res['contribution_eur'].sum():,.0f})"]
    ws = res.attrs.get("waiting_stats", {})
    if base.exit_after_months > 1:
        L.append(f"Waiting period: {ws.get('recovered', 0)} times a stock came back inside the top {base.buffer} x N "
                 f"while waiting (kept); {ws.get('sold_after_wait', 0)} times it was sold after waiting.")

    L += ["", "Section 10.3 - minimum acceptable (in-sample):"]
    checks = [("Net excess return above +2% a year", s["excess"] > 0.02, pct(s["excess"])),
              ("Sharpe above the S&P 500's", s["sharpe"] > s["bench_sharpe"], f"{s['sharpe']:.2f} vs {s['bench_sharpe']:.2f}"),
              ("Information ratio above 0.4", s["ir"] > 0.4, f"{s['ir']:.2f}"),
              ("Worst fall smaller than the S&P 500's", s["max_dd"] > s["bench_max_dd"], f"{pct(s['max_dd'])} vs {pct(s['bench_max_dd'])}"),
              ("Beats the S&P 500 in more than 50% of months", s["hit"] > 0.5, f"{s['hit']:.0%}"),
              ("Turnover below 150% a year", to < 1.5, f"{to:.0%}"),
              ("Deflated Sharpe above 0.90", dsr > 0.90, f"{dsr:.2f} ({len(trial_srs)} trial(s) logged)")]
    L += [f"  [{'PASS' if ok else 'FAIL'}] {name:<46} {val}" for name, ok, val in checks]
    L.append(f"  [info] Factor-adjusted alpha (FF5 + momentum, USD): {reg.loc['alpha', 'coef'] * 12:+.1%} a year, "
             f"t-stat {reg.loc['alpha', 't']:.2f}")

    L += ["", "Section 10.4 - robustness:", "  Sub-periods (net excess return a year):"]
    sub_pos = 0
    for lab, a, z in SUB_PERIODS:
        rr, bb = r.loc[a:z], b.loc[a:z]
        ex = mt.cagr(rr) - mt.cagr(bb)
        sub_pos += ex > 0
        L.append(f"    {lab:<16}{pct(ex):>8}   (strategy {pct(mt.cagr(rr))}, S&P 500 {pct(mt.cagr(bb))})")
    L.append(f"  [{'PASS' if sub_pos >= 2 else 'FAIL'}] Positive in at least 2 of 3 sub-periods ({sub_pos} of 3)")

    L += ["", f"  {'Variant':<32}{'Return':>9}{'Excess':>9}{'IR':>7}{'Turnover':>10}"]
    L.append(f"  {'Base case':<32}{pct(s['cagr']):>9}{pct(s['excess']):>9}{s['ir']:>7.2f}{to:>10.0%}")
    var = {}
    for cfg in robustness_configs(base):
        print(f"  {cfg.label}...", flush=True)
        vr, _ = mkt.run(cfg)
        vs = mt.summary(vr["ret_eur"], vr["bench_eur"], rf.reindex(vr.index).fillna(0.0))
        var[cfg.label] = vs
        L.append(f"  {cfg.label:<32}{pct(vs['cagr']):>9}{pct(vs['excess']):>9}{vs['ir']:>7.2f}{turnover(vr):>10.0%}")
    hold_ok = all(var[f"{n} holdings throughout"]["excess"] > 0 for n in (25, 40, 50))
    cost_ok = var["2x trading costs"]["excess"] > 0
    loo_ok = all(var[f"Without {k}"]["excess"] > 0 for k in WEIGHTS)
    shift = var["Trade 5 trading days later"]["excess"]
    shift_ok = s["excess"] > 0 and shift > 0.7 * s["excess"]
    L += [f"  [{'PASS' if hold_ok else 'FAIL'}] 25, 40 and 50 holdings all beat the S&P 500",
          f"  [{'PASS' if cost_ok else 'FAIL'}] Still beats the S&P 500 at 2x costs",
          f"  [{'PASS' if loo_ok else 'FAIL'}] No single score carries the result (beats the S&P 500 without each one)",
          f"  [{'PASS' if shift_ok else 'FAIL'}] Trading 5 days later keeps at least 70% of the excess return",
          "  [later] Parameter sensitivity (look-back lengths) and probability of backtest overfitting: Stage 05b"]

    L += ["", "Calendar years (EUR, net):", f"  {'Year':<6}{'Strategy':>10}{'S&P 500':>10}{'Excess':>10}"]
    for y, g in res.groupby(res.index.year):
        ry, by = (1 + g["ret_eur"]).prod() - 1, (1 + g["bench_eur"]).prod() - 1
        part = "" if len(g) == 12 else f"  ({len(g)} months)"
        L.append(f"  {y:<6}{pct(ry):>10}{pct(by):>10}{pct(ry - by):>10}{part}")

    L += ["", f"Factor exposures (FF5 + momentum, monthly USD excess returns, R2 {reg.attrs['r2']:.2f}):"]
    for k, row in reg.iterrows():
        v = row["coef"] * 12 if k == "alpha" else row["coef"]
        L.append(f"  {k:<8}{v:>+8.2f}{'':<2}t {row['t']:>6.2f}" + ("   (a year)" if k == "alpha" else ""))
    L += ["", f"Deflated Sharpe: {dsr:.2f}, using {len(trial_srs)} logged trial(s); the monthly Sharpe bar was {bar:.3f}.",
          "Every new setup you run is added to reports/trial_log.csv and raises this bar.",
          "", "Notes: the S&P 500 is SPY (fees included). Sharpe uses the US T-bill rate. Factor months are calendar months,",
          "while holding periods run from the 2nd trading day to the 2nd trading day, so exposures are approximate.",
          "", f"Files: {MONTHLY_PATH.name}, {TRADES_PATH.name}, {CHART_PATH.name}, {TRIAL_LOG.name}"]
    REPORT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\nReport written to {REPORT_PATH.relative_to(sh.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
