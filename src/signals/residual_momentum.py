"""Residual momentum (Design Doc 6.1.1; Blitz, Huij & Martens, 2011).

For a formation month-end F (trades happen in month F+1):
  1. Regress each stock's monthly excess return on the Fama-French 3 factors over the
     36 months F-36 .. F-1. The window ends one month before F, so the live version
     only needs factor data French has already published.
  2. Keep the residuals (the part not explained by market, size and value) for the
     11 months F-11 .. F-1, i.e. t-12 .. t-2. The latest month is skipped, as in all
     momentum measures, because one-month returns tend to reverse.
  3. Score = sum of those residuals / their standard deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EST_MONTHS = 36
MIN_EST_MONTHS = 30
RANK_MONTHS = 11
MIN_RANK_MONTHS = 9
FACTORS = ["Mkt-RF", "SMB", "HML"]


def _score_from_residuals(res: np.ndarray) -> float:
    r = res[-RANK_MONTHS:]
    r = r[~np.isnan(r)]
    if len(r) < MIN_RANK_MONTHS:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.sum() / sd) if sd > 0 else np.nan


def residual_momentum_at(excess: pd.DataFrame, factors: pd.DataFrame, formation: pd.Timestamp,
                         tickers: list[str]) -> pd.Series:
    """Residual-momentum score at `formation` for `tickers`.

    `excess`: wide monthly excess returns (month-ends x tickers).
    `factors`: monthly factor returns with columns FACTORS, same month-end index.
    """
    if formation not in excess.index:
        return pd.Series(np.nan, index=tickers, dtype="float64")
    p = excess.index.get_loc(formation)
    if p < EST_MONTHS:
        return pd.Series(np.nan, index=tickers, dtype="float64")
    window = excess.index[p - EST_MONTHS:p]                      # F-36 .. F-1
    X = factors.reindex(window)[FACTORS]
    if X.isna().any(axis=None):
        return pd.Series(np.nan, index=tickers, dtype="float64")
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    Y = excess.reindex(index=window, columns=tickers).to_numpy(dtype="float64")

    out = np.full(len(tickers), np.nan)
    complete = ~np.isnan(Y).any(axis=0)
    if complete.any():                                           # one solve for all full histories
        beta, *_ = np.linalg.lstsq(X, Y[:, complete], rcond=None)
        res = Y[:, complete] - X @ beta
        out[complete] = [_score_from_residuals(res[:, j]) for j in range(res.shape[1])]
    for j in np.flatnonzero(~complete):                          # stocks with a few gaps
        ok = ~np.isnan(Y[:, j])
        if ok.sum() < MIN_EST_MONTHS:
            continue
        beta, *_ = np.linalg.lstsq(X[ok], Y[ok, j], rcond=None)
        res = np.full(len(window), np.nan)
        res[ok] = Y[ok, j] - X[ok] @ beta
        out[j] = _score_from_residuals(res)
    return pd.Series(out, index=tickers)
