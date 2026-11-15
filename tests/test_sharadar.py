"""Tests for the Sharadar loader, using small made-up tables in Sharadar's layout."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import src.data.sharadar as sh


# --------------------------------------------------------------------------- #
# Fixtures: tiny versions of each Sharadar table
# --------------------------------------------------------------------------- #
def tickers_raw():
    return pd.DataFrame({
        "table": ["SEP", "SF1", "SEP", "SEP", "SEP"],
        "permaticker": [1, 1, 2, 3, 4],
        "ticker": ["AAA", "AAA", "BBB1", "CCC", "ETFX"],
        "name": ["Alpha", "Alpha", "Beta (old)", "Gamma", "Some ADR"],
        "exchange": ["NYSE"] * 5,
        "isdelisted": ["N", "N", "Y", "Y", "N"],
        "category": ["Domestic Common Stock", "Domestic Common Stock",
                     "Domestic Common Stock Primary Class", "Domestic Common Stock", "ADR Common Stock"],
        "sector": ["Tech"] * 5,
        "firstpricedate": ["1998-01-02"] * 5,
        "lastpricedate": ["2026-10-30", "2026-10-30", "2001-03-15", "2002-06-28", "2026-10-30"],
    })


def daily_prices():
    """AAA trades throughout; BBB1 stops mid-March 2001 (bankrupt); CCC stops June 2002 (acquired)."""
    rows = []
    for t, end in [("AAA", "2002-12-31"), ("BBB1", "2001-03-15"), ("CCC", "2002-06-28")]:
        days = pd.bdate_range("2001-01-01", end)
        price = 100 * (1.001 ** np.arange(len(days)))
        rows.append(pd.DataFrame({"ticker": t, "date": days, "close": price, "closeadj": price,
                                  "closeunadj": price, "volume": 1_000.0}))
    return pd.concat(rows, ignore_index=True)


def actions_raw():
    return pd.DataFrame({
        "date": ["2001-03-20", "2002-07-01", "2002-07-01"],
        "action": ["bankruptcyliquidation", "acquisitionby", "delisted"],
        "ticker": ["BBB1", "CCC", "CCC"],
        "value": [np.nan] * 3,
        "contraticker": [None, "AAA", None],
    })


def sp500_raw():
    return pd.DataFrame({
        "date": ["2001-01-31", "2001-01-31", "2001-03-16", "2001-03-16", "2002-06-29", "2002-06-29"],
        "action": ["historical", "historical", "removed", "added", "removed", "added"],
        "ticker": ["AAA", "BBB1", "BBB1", "CCC", "CCC", "DDD"],
    })


def fundamentals_raw():
    return pd.DataFrame({
        "ticker": ["AAA"] * 4,
        "dimension": ["ARQ", "ART", "MRQ", "ARQ"],
        "calendardate": ["2001-03-31"] * 3 + ["2001-06-30"],
        "datekey": ["2001-05-10", "2001-05-10", "2001-05-10", "2001-08-31"],
        "reportperiod": ["2001-03-31"] * 3 + ["2001-06-30"],
        "revenue": [10.0, 40.0, 10.5, 11.0],
        "equity": [50.0, 50.0, 49.0, 52.0],
        "assets": [100.0, 100.0, 100.0, 104.0],
    })


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nSHARADAR_API_KEY="abc123"\n', encoding="utf-8")
    assert sh.load_api_key(env) == "abc123"
    monkeypatch.setenv("SHARADAR_API_KEY", "fromenv")
    assert sh.load_api_key(env) == "fromenv"


def test_api_key_missing_gives_clear_message(tmp_path, monkeypatch):
    monkeypatch.delenv("SHARADAR_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="SHARADAR_API_KEY"):
        sh.load_api_key(tmp_path / ".env")


def test_old_column_names_are_mapped():
    df = sh.normalize_columns(pd.DataFrame({"Ticker": ["A"], "DATEKEY": ["2001-01-01"]}))
    assert list(df.columns) == ["ticker", "date"]


def test_tickers_keep_only_us_common_stocks_including_delisted():
    t = sh.build_tickers(tickers_raw())
    assert set(t["ticker"]) == {"AAA", "BBB1", "CCC"}  # ADR and fundamentals row dropped
    assert t.set_index("ticker").loc["BBB1", "isdelisted"] == "Y"


def test_sp500_membership_rebuilt_from_snapshot_and_events():
    dates = pd.DatetimeIndex(["2000-12-31", "2001-01-31", "2001-02-28", "2001-03-31", "2002-07-31"])
    m = sh.build_sp500_membership(sp500_raw(), dates)
    on = lambda d: set(m.loc[m["date"] == d, "ticker"])
    assert on("2000-12-31") == set()                 # before any snapshot
    assert on("2001-01-31") == {"AAA", "BBB1"}       # snapshot
    assert on("2001-02-28") == {"AAA", "BBB1"}
    assert on("2001-03-31") == {"AAA", "CCC"}        # BBB1 replaced by CCC
    assert on("2002-07-31") == {"AAA", "DDD"}        # CCC replaced by DDD


def test_monthly_returns_and_gaps():
    daily = daily_prices()
    m = sh.build_monthly(daily)
    aaa = m[m["ticker"] == "AAA"].set_index("date")
    jan, feb = daily[daily.ticker.eq("AAA") & daily.date.dt.month.eq(1) & daily.date.dt.year.eq(2001)], \
               daily[daily.ticker.eq("AAA") & daily.date.dt.month.eq(2) & daily.date.dt.year.eq(2001)]
    expected = feb["closeadj"].iloc[-1] / jan["closeadj"].iloc[-1] - 1
    assert aaa.loc["2001-02-28", "ret"] == pytest.approx(expected, rel=1e-6)
    assert np.isnan(aaa.loc["2001-01-31", "ret"])    # first month has no prior price
    assert aaa.loc["2001-01-31", "adv_usd"] > 0

    gappy = daily[~((daily.ticker == "AAA") & (daily.date.dt.month == 5) & (daily.date.dt.year == 2001))]
    g = sh.build_monthly(gappy).query("ticker == 'AAA'").set_index("date")
    assert np.isnan(g.loc["2001-06-30", "ret"])      # no return across a missing month


def test_delisting_return_only_for_cause():
    monthly = sh.build_monthly(daily_prices())
    out, kinds = sh.add_delisting_returns(monthly, sh.build_tickers(tickers_raw()), actions_raw(),
                                          data_end=pd.Timestamp("2002-12-31"))
    added = out[out["delisting_adjusted"]]
    assert list(added["ticker"]) == ["BBB1"]
    assert added["date"].iloc[0] == pd.Timestamp("2001-04-30")
    assert added["ret"].iloc[0] == pytest.approx(-0.30)
    assert kinds["bankruptcyliquidation"] == 1 and kinds["acquisitionby"] == 1


def test_fundamentals_as_reported_only():
    f, missing = sh.build_fundamentals(sh.normalize_columns(fundamentals_raw()))
    assert set(f["dimension"]) == {"ARQ", "ART"}     # restated MRQ excluded
    assert "filed" in f.columns and "cor" in missing


def test_fundamentals_missing_required_column():
    with pytest.raises(SystemExit, match="equity"):
        sh.build_fundamentals(sh.normalize_columns(fundamentals_raw().drop(columns="equity")))


def test_fundamentals_asof_never_uses_same_day_or_future_filings():
    f, _ = sh.build_fundamentals(sh.normalize_columns(fundamentals_raw()))
    panel = pd.DataFrame({"date": pd.to_datetime(["2001-05-10", "2001-05-11", "2001-08-31", "2001-09-30"]),
                          "ticker": "AAA"})
    a = sh.fundamentals_asof(f, panel, "ARQ").set_index("date")
    assert np.isnan(a.loc["2001-05-10", "equity"])   # filed that day: not yet usable
    assert a.loc["2001-05-11", "equity"] == 50.0
    assert a.loc["2001-08-31", "equity"] == 50.0     # Q2 filing that same day not used yet
    assert a.loc["2001-09-30", "equity"] == 52.0


def _zip(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("table.csv", df.to_csv(index=False))
    return buf.getvalue()


def test_read_table_selects_columns_and_aliases(tmp_path):
    p = tmp_path / "fundamentals_20261106.zip"
    p.write_bytes(_zip(fundamentals_raw()))
    df = sh.read_table(p, usecols=["ticker", "date", "equity"])
    assert list(df.columns) == ["ticker", "date", "equity"]


def test_full_build_end_to_end(tmp_path, monkeypatch):
    """Runs the whole build on the tiny tables and checks the report is written."""
    pytest.importorskip("pyarrow")
    raw, clean, report = tmp_path / "raw", tmp_path / "clean", tmp_path / "reports" / "qa.txt"
    raw.mkdir()
    monkeypatch.setattr(sh, "RAW_DIR", raw)
    monkeypatch.setattr(sh, "CLEAN_DIR", clean)
    monkeypatch.setattr(sh, "REPORT_PATH", report)
    monkeypatch.setattr(sh, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sh, "START", pd.Timestamp("2001-01-01"))
    stocks = daily_prices().assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
    for name, df in [("tickers", tickers_raw()), ("actions", actions_raw()), ("sp500", sp500_raw()),
                     ("fundamentals", fundamentals_raw()), ("stocks", stocks)]:
        (raw / f"{name}_20261106.zip").write_bytes(_zip(df))

    code = sh.build_all()
    text = report.read_text(encoding="utf-8")
    assert "S&P 500 members per month-end" in text
    assert code == 1                                  # 2-member toy index fails the 480-520 count check
    assert (clean / "monthly.parquet").exists()


def test_filings_before_period_end_are_dropped():
    raw = fundamentals_raw()
    raw.loc[3, "datekey"] = "2001-06-15"            # "filed" before the June quarter ended
    f, _ = sh.build_fundamentals(sh.normalize_columns(raw))
    assert f.attrs["dropped_early_filings"] == 1
    assert (f["filed"] >= f["reportperiod"]).all()


def test_monthly_works_with_category_tickers():
    daily = daily_prices()
    a = sh.build_monthly(daily)
    b = sh.build_monthly(daily.assign(ticker=daily["ticker"].astype("category")))
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_qa_separates_data_spikes_from_real_moves():
    days = pd.bdate_range("2001-03-01", "2001-03-30")
    spike = np.full(len(days), 100.0); spike[10] = 200.0            # one-day spike: data error
    jump = np.full(len(days), 100.0); jump[10:] = 160.0              # lasting move: real event
    daily = pd.concat([pd.DataFrame({"ticker": t, "date": days, "close": p, "closeadj": p,
                                     "closeunadj": p, "volume": 1e6}) for t, p in [("SPK", spike), ("JMP", jump)]])
    monthly = sh.build_monthly(daily)
    membership = pd.DataFrame({"date": pd.Timestamp("2001-03-31"), "ticker": ["SPK", "JMP"]})
    fund, _ = sh.build_fundamentals(sh.normalize_columns(fundamentals_raw()))
    tickers = pd.DataFrame({"ticker": ["SPK", "JMP"], "isdelisted": ["N", "N"]})
    summary, fails, warns = sh.qa_checks(daily, monthly, membership, fund, tickers)
    text = " ".join(summary + warns)
    assert "1 lasting, 2 reversed" in text            # both legs of the spike
    assert any("SPK 2001-03-15" in w for w in warns)
    assert any("JMP 2001-03-15" in s for s in summary)


def test_fresh_step_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "CLEAN_DIR", tmp_path)
    src, out = tmp_path / "in.zip", tmp_path / "out.parquet"
    src.write_text("x"); out.write_text("y")
    import os, time
    os.utime(src, (time.time() - 100, time.time() - 100))
    assert sh.is_fresh("daily", out, [src])           # newer output, no record -> version 1
    assert not sh.is_fresh("fundamentals", out, [src])  # fundamentals now at version 2
    sh._record("fundamentals")
    assert sh.is_fresh("fundamentals", out, [src])


def test_dates_in_different_precisions_still_join():
    """Parquet returns nanoseconds while pandas 3 parses text dates as microseconds."""
    f, _ = sh.build_fundamentals(sh.normalize_columns(fundamentals_raw()))
    f["filed"] = f["filed"].astype("datetime64[us]")
    panel = pd.DataFrame({"date": pd.to_datetime(["2001-09-30"]).astype("datetime64[ns]"), "ticker": "AAA"})
    assert sh.fundamentals_asof(f, panel, "ARQ")["equity"].iloc[0] == 52.0
