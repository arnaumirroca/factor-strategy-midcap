"""Build the four stock scores and the composite for every month-end (Stage 04).

Run from the project folder, after the universe is built (python -m src.data.size_universe):

    .\\.venv\\Scripts\\python.exe -m src.signals.build

Output
------
data/clean/signals/scores.parquet    One row per month-end and stock in the universe
                                     (US stocks ranked 501-1,500 by market value): raw characteristics, the four scores
                                     (-1 worst .. +1 best), composite and rank.
reports/signals_report.txt           Coverage, top-10 lists to check by eye, and how
                                     well each score ranked next month's returns.
reports/signals_top25_samples.csv    Top 25 at the sample dates, to open in Excel.

The hold-out rule (Design Doc 10.1)
-----------------------------------
Scores are computed for every month, but anything that looks at what happened
AFTER a score was computed (the IC and the top-minus-bottom spread) only uses
holding months up to December 2015. 2016 onwards is the hold-out for Gate 2 and is
never shown here.
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from src.data import sharadar as sh
from src.data.size_universe import OUT_PATH as UNIVERSE_PATH
from src.data.gate1_momentum import split_adjusted_shares
from src.signals import characteristics as ch
from src.signals.common import WEIGHTS, combine, mean_of_available, rank_score, spearman_by_date
from src.signals.residual_momentum import FACTORS, residual_momentum_at

IN_SAMPLE_END = pd.Timestamp("2015-12-31")   # last holding month any outcome may come from
MIN_PRICE = 5.0                              # Section 4.1
MIN_ADV_USD = 2e6                            # Section 4.1 (mid/small caps): orders stay far below 1% of volume
MIN_HISTORY_MONTHS = 30                      # returns in the last 36 months
STALE_FILING_DAYS = 400                      # older filings are treated as missing
SAMPLE_DATES = ["2000-12-31", "2008-12-31", "2015-12-31"]
SCORES = list(WEIGHTS)

OUT_DIR = sh.CLEAN_DIR.parent / "signals"
OUT_PATH = OUT_DIR / "scores.parquet"
REPORT_PATH = sh.PROJECT_ROOT / "reports" / "signals_report.txt"
SAMPLES_PATH = sh.PROJECT_ROOT / "reports" / "signals_top25_samples.csv"


def _long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    s = wide.stack()
    s.index.names = ["date", "ticker"]
    return s.rename(name).dropna().reset_index()


def load_inputs():
    clean = sh.CLEAN_DIR
    need = ["monthly", "tickers", "fundamentals", "prices_daily"]
    for n in need:
        if not (clean / f"{n}.parquet").exists():
            raise SystemExit(f"Missing {n}.parquet. Copy the cleaned data first (README, step 2).")
    if not UNIVERSE_PATH.exists():
        raise SystemExit("Missing the universe. Run: python -m src.data.size_universe")
    membership = sh.read_parquet(UNIVERSE_PATH, columns=["date", "ticker"])
    ever = set(membership["ticker"])
    monthly = sh.read_parquet(clean / "monthly.parquet", columns=["date", "ticker", "ret", "price", "adv_usd"])
    monthly = monthly[monthly["ticker"].isin(ever)]
    tickers = sh.read_parquet(clean / "tickers.parquet", columns=["ticker", "name", "sector"])
    fund = sh.read_parquet(clean / "fundamentals.parquet")
    fund = fund[fund["ticker"].isin(ever)]
    daily = sh.read_parquet(clean / "prices_daily.parquet", columns=["ticker", "date", "close", "closeadj", "closeunadj"],
                            read_dictionary=["ticker"])
    daily = daily[daily["ticker"].isin(ever)].copy()
    daily["ticker"] = daily["ticker"].astype(str)
    ff3 = sh.read_parquet(sh.PROJECT_ROOT / "data" / "clean" / "fama_french" / "us_ff3.parquet")
    if "date" in ff3.columns:
        ff3 = ff3.set_index("date")
    ff3.index = pd.to_datetime(ff3.index).astype("datetime64[ns]")
    for df in (membership, monthly, daily):
        df["date"] = df["date"].astype("datetime64[ns]")
    return membership, monthly, tickers, fund, daily, ff3


def build_panel(membership, monthly, tickers, fund, daily, ff3) -> pd.DataFrame:
    """Raw characteristics for every universe member at every month-end, with the price, volume and history filters."""
    R = monthly.pivot_table(index="date", columns="ticker", values="ret", aggfunc="last")
    months = R.index
    P = monthly.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").reindex_like(R)
    ADV = monthly.pivot_table(index="date", columns="ticker", values="adv_usd", aggfunc="last").reindex_like(R)
    HIST = R.notna().rolling(36, min_periods=1).sum()

    print("  daily volatility and split factors...", flush=True)
    close_adj = daily.pivot(index="date", columns="ticker", values="closeadj")
    vol = ch.at_month_ends(ch.daily_volatility(close_adj.pct_change(fill_method=None)), months)
    ratio = (daily.pivot(index="date", columns="ticker", values="closeunadj")
             / daily.pivot(index="date", columns="ticker", values="close"))
    K = ch.at_month_ends(ratio.ffill(limit=5), months)

    panel = membership[membership["date"].isin(months)][["date", "ticker"]].copy()
    for wide, name in [(P, "price"), (ADV, "adv_usd"), (HIST, "history"), (vol, "vol"), (K, "k")]:
        panel = panel.merge(_long(wide.reindex(index=months), name), on=["date", "ticker"], how="left")
    panel["in_universe"] = ((panel["price"] > MIN_PRICE) & (panel["adv_usd"] > MIN_ADV_USD)
                            & (panel["history"] >= MIN_HISTORY_MONTHS))
    panel = panel.merge(tickers, on="ticker", how="left")
    panel["sector"] = panel["sector"].fillna("Unknown")

    print("  matching the latest filing before each month-end...", flush=True)
    q = sh.fundamentals_asof(fund[["ticker", "dimension", "filed", "equity", "assets", "debt", "sharesbas"]], panel, "ARQ")
    t = sh.fundamentals_asof(fund[["ticker", "dimension", "filed", "revenue", "cor", "sgna", "intexp", "netinccmn", "fcf"]],
                             panel, "ART")
    q_cols, t_cols = ["equity", "assets", "debt", "sharesbas"], ["revenue", "cor", "sgna", "intexp", "netinccmn", "fcf"]
    for df, cols in ((q, q_cols), (t, t_cols)):
        stale = (df["date"] - df["filed"]).dt.days > STALE_FILING_DAYS
        df.loc[stale, cols] = np.nan
    panel = panel.merge(q[["date", "ticker", "filed"] + q_cols], on=["date", "ticker"], how="left")
    panel = panel.merge(t[["date", "ticker"] + t_cols], on=["date", "ticker"], how="left")

    panel["filed_month"] = (panel["filed"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    k_long = _long(K, "k_filed").rename(columns={"date": "filed_month"})
    panel = panel.merge(k_long, on=["filed_month", "ticker"], how="left")
    panel["shares"] = split_adjusted_shares(panel["sharesbas"], panel["k_filed"].fillna(panel["k"]), panel["k"])
    panel["market_cap"] = panel["price"] * panel["shares"]

    panel["low_vol_raw"] = ch.low_vol_raw(panel["vol"])
    panel["op"] = ch.operating_profitability(panel["revenue"], panel["cor"], panel["sgna"], panel["intexp"], panel["equity"])
    panel["leverage_raw"] = ch.low_leverage(panel["debt"], panel["assets"])
    panel = pd.concat([panel, ch.value_ratios(panel["equity"], panel["netinccmn"], panel["fcf"], panel["market_cap"])], axis=1)

    print("  residual momentum regressions...", flush=True)
    rf = ff3["RF"].reindex(months)
    EX = R.sub(rf, axis=0)
    fac = ff3[FACTORS].reindex(months)
    parts = []
    for F, g in panel[panel["in_universe"]].groupby("date"):
        s = residual_momentum_at(EX, fac, F, list(g["ticker"]))
        parts.append(pd.DataFrame({"date": F, "ticker": s.index, "res_mom_raw": s.to_numpy()}))
    panel = panel.merge(pd.concat(parts), on=["date", "ticker"], how="left")
    return panel


def score_month(g: pd.DataFrame) -> pd.DataFrame:
    """Turn one month's raw characteristics into the four scores and the composite."""
    sec = g["sector"]
    out = pd.DataFrame(index=g.index)
    out["low_vol"] = rank_score(g["low_vol_raw"])                                    # whole universe
    quality = mean_of_available([rank_score(g["op"], sec), rank_score(g["leverage_raw"], sec)], 1)
    out["quality"] = rank_score(quality.where(g["op"].notna()), sec)
    value = mean_of_available([rank_score(g[c], sec) for c in ("bm", "ep", "fcfp")], 2)
    out["value"] = rank_score(value, sec)
    out["res_mom"] = rank_score(g["res_mom_raw"], sec)
    out["composite"] = combine(out)
    out["rank"] = out["composite"].rank(ascending=False, method="first")
    return out


def build_scores(panel: pd.DataFrame) -> pd.DataFrame:
    uni = panel[panel["in_universe"]].copy()
    scored = pd.concat([score_month(g) for _, g in uni.groupby("date")])
    keep = ["date", "ticker", "name", "sector", "price", "market_cap", "vol", "op", "leverage_raw",
            "bm", "ep", "fcfp", "res_mom_raw"]
    return uni[keep].join(scored).sort_values(["date", "rank"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def in_sample_outcomes(scores: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Attach next month's return, keeping only holding months up to IN_SAMPLE_END."""
    nxt = monthly[["date", "ticker", "ret"]].copy()
    nxt["date"] = (nxt["date"] - pd.offsets.MonthEnd(1)).astype("datetime64[ns]")  # return of F+1, keyed at F
    df = scores.merge(nxt.rename(columns={"ret": "next_ret"}), on=["date", "ticker"], how="left")
    holding_month = (df["date"] + pd.offsets.MonthEnd(1)).astype("datetime64[ns]")
    return df[holding_month <= IN_SAMPLE_END]


def report(scores: pd.DataFrame, panel: pd.DataFrame, monthly: pd.DataFrame) -> list[str]:
    lines = [f"Signal report - {date.today():%d %b %Y}", "",
             f"Month-ends scored: {scores['date'].nunique()} ({scores['date'].min():%Y-%m} to {scores['date'].max():%Y-%m})"]
    size = scores.groupby("date").size()
    members = panel.groupby("date").size()
    lines.append(f"Stocks passing the universe filters per month: min {size.min()}, median {int(size.median())}, "
                 f"max {size.max()} (of about {int(members.median())} stocks ranked 501-1,500)")
    lines += ["", "Coverage (share of universe with each score):"]
    for c in SCORES + ["composite"]:
        lines.append(f"  {c:<10} {scores[c].notna().mean():6.1%}")

    lines += ["", "Top 10 by composite score at sample dates (check these make sense):"]
    samples = []
    for d in SAMPLE_DATES:
        top = scores[scores["date"] == pd.Timestamp(d)].nsmallest(25, "rank")
        if top.empty:
            continue
        samples.append(top)
        lines.append(f"  {d}")
        lines.append(f"    {'#':>2}  {'Ticker':<7}{'Name':<28}{'Sector':<24}{'Comp':>6}{'LowV':>6}{'Qual':>6}{'RMom':>6}{'Val':>6}")
        for _, r in top.head(10).iterrows():
            f = lambda x: "   -  " if pd.isna(x) else f"{x:+6.2f}"
            lines.append(f"    {int(r['rank']):>2}  {r['ticker']:<7}{str(r['name'])[:27]:<28}{str(r['sector'])[:23]:<24}"
                         f"{f(r['composite'])}{f(r['low_vol'])}{f(r['quality'])}{f(r['res_mom'])}{f(r['value'])}")
    if samples:
        SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(samples).to_csv(SAMPLES_PATH, index=False, float_format="%.4f")

    df = in_sample_outcomes(scores, monthly)
    n_months = df["date"].nunique()
    lines += ["", f"How well each score ranked next month's returns - IN-SAMPLE ONLY "
                  f"(holding months {df['date'].min() + pd.offsets.MonthEnd(1):%Y-%m} to "
                  f"{df['date'].max() + pd.offsets.MonthEnd(1):%Y-%m}, {n_months} months)",
              "  IC = rank correlation between the score and next month's return, averaged over months.",
              "  Positive = the score pointed the right way. Typical good single signals: +0.02 to +0.05.", "",
              f"  {'Score':<10}{'Mean IC':>9}{'t-stat':>8}{'IC > 0':>8}{'Top-bottom 20% (a year)':>26}"]
    for c in SCORES + ["composite"]:
        ic = spearman_by_date(df, c, "next_ret").dropna()
        t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic))) if len(ic) > 2 else np.nan

        def spread(g):
            g = g[[c, "next_ret"]].dropna()
            if len(g) < 50:
                return np.nan
            lo, hi = g[c].quantile([0.2, 0.8])
            return g.loc[g[c] >= hi, "next_ret"].mean() - g.loc[g[c] <= lo, "next_ret"].mean()
        sp = df.groupby("date")[[c, "next_ret"]].apply(spread).mean() * 12
        lines.append(f"  {c:<10}{ic.mean():>+9.3f}{t:>8.2f}{(ic > 0).mean():>8.0%}{sp:>+25.1%}")

    lines += ["", "Average correlation between the scores (in-sample months):",
              "  " + " " * 10 + "".join(f"{c:>10}" for c in SCORES)]
    ins = scores[scores["date"] <= IN_SAMPLE_END]
    corr = ins.groupby("date")[SCORES].apply(lambda g: g.rank().corr()).groupby(level=1).mean().reindex(index=SCORES, columns=SCORES)
    for c in SCORES:
        lines.append(f"  {c:<10}" + "".join(f"{corr.loc[c, k]:>10.2f}" for k in SCORES))

    lines += ["", "Hold-out: nothing after Dec 2015 is shown. It is reserved for Gate 2.",
              f"Top 25 at the sample dates: {SAMPLES_PATH.relative_to(sh.PROJECT_ROOT)}"]
    return lines


def run() -> int:
    print("Stage 04: building the four scores")
    membership, monthly, tickers, fund, daily, ff3 = load_inputs()
    panel = build_panel(membership, monthly, tickers, fund, daily, ff3)
    del daily
    scores = build_scores(panel)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sh.write_parquet(scores, OUT_PATH)
    print(f"  saved {len(scores):,} rows to {OUT_PATH.relative_to(sh.PROJECT_ROOT)}")
    lines = report(scores, panel, monthly)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nReport written to {REPORT_PATH.relative_to(sh.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
