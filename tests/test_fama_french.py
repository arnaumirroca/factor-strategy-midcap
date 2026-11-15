"""Tests for the Fama-French parser, using small samples in the library's real layout."""

import io
import zipfile

import pandas as pd
import pytest

from src.data.fama_french import parse_monthly_section, quality_checks, read_zip

FF3_SAMPLE = (
    "This file was created by CMPT_ME_BEME_RETS using the 202509 CRSP database.\r\n"
    "The 1-month TBill return is from Ibbotson and Associates, Inc.\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "192607,    2.89,   -2.55,   -2.39,    0.22\r\n"
    "192608,    2.64,   -1.14,    3.81,    0.25\r\n"
    "192609,    0.38,   -1.36,    0.05,    0.23\r\n"
    "\r\n"
    " Annual Factors: January-December \r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "1927,   29.47,   -2.04,   -4.54,    3.12\r\n"
)

MOM_SAMPLE = (
    "This file was created using the 202509 CRSP database.\n"
    "\n"
    ",Mom   \n"
    "192701,    0.36\n"
    "192702,  -99.99\n"
    "\n"
    "Annual Factors:\n"
    ",Mom\n"
    "1928,   26.20\n"
)


def test_parses_monthly_block_only():
    df = parse_monthly_section(FF3_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3  # annual rows excluded
    assert df.index[0] == pd.Timestamp("1926-07-31")
    assert df.loc["1926-07-31", "Mkt-RF"] == pytest.approx(0.0289)  # percent -> decimal


def test_momentum_renamed_and_missing_code_handled():
    df = parse_monthly_section(MOM_SAMPLE)
    assert list(df.columns) == ["MOM"]
    assert pd.isna(df.loc["1927-02-28", "MOM"])  # -99.99 means missing


def test_read_zip_returns_csv_text():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_Factors.CSV", FF3_SAMPLE)
    assert "Mkt-RF" in read_zip(buf.getvalue())


def test_quality_checks_flag_gaps_and_missing():
    assert quality_checks("ff3", parse_monthly_section(FF3_SAMPLE)) == []
    problems = quality_checks("mom", parse_monthly_section(MOM_SAMPLE))
    assert any("missing values" in p for p in problems)

    gappy = parse_monthly_section(FF3_SAMPLE).drop(pd.Timestamp("1926-08-31"))
    assert any("missing months" in p for p in quality_checks("gap", gappy))


def test_raises_on_unexpected_file():
    with pytest.raises(ValueError):
        parse_monthly_section("no data here\n,just,text\n")
