"""Performance measures (Design Doc 10.3)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

N = NormalDist()
EULER_GAMMA = 0.5772156649


def cagr(r: pd.Series) -> float:
    return float((1 + r).prod() ** (12 / len(r)) - 1) if len(r) else np.nan


def volatility(r: pd.Series) -> float:
    return float(r.std(ddof=1) * math.sqrt(12))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = r - rf
    return float(ex.mean() / ex.std(ddof=1) * math.sqrt(12))


def drawdowns(r: pd.Series) -> pd.Series:
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdowns(r).min())


def relative_drawdown(r: pd.Series, b: pd.Series) -> float:
    rel = (1 + r).cumprod() / (1 + b).cumprod()
    return float((rel / rel.cummax() - 1).min())


def tracking_error(r: pd.Series, b: pd.Series) -> float:
    return volatility(r - b)


def information_ratio(r: pd.Series, b: pd.Series) -> float:
    return float((cagr(r) - cagr(b)) / tracking_error(r, b))


def beta(r: pd.Series, b: pd.Series) -> float:
    return float(np.cov(r, b, ddof=1)[0, 1] / np.var(b, ddof=1))


def hit_rate(r: pd.Series, b: pd.Series) -> float:
    return float((r > b).mean())


def ols(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Plain least squares with a constant: coefficient, t-stat (no extra dependencies)."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(df)), df[X.columns].to_numpy()])
    yv = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ coef
    dof = len(df) - A.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * (resid @ resid) / dof)
    out = pd.DataFrame({"coef": coef, "t": coef / se}, index=["alpha"] + list(X.columns))
    out.attrs["r2"] = 1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean()))
    out.attrs["months"] = len(df)
    return out


def probabilistic_sharpe(excess: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Probability that the true Sharpe is above `sr_benchmark` (monthly units)."""
    x = excess.dropna()
    t = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3 = float(((x - x.mean()) ** 3).mean() / x.std(ddof=0) ** 3)
    g4 = float(((x - x.mean()) ** 4).mean() / x.std(ddof=0) ** 4)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return N.cdf((sr - sr_benchmark) * math.sqrt(t - 1) / denom)


def deflated_sharpe(excess: pd.Series, trial_sharpes_monthly: list[float]) -> tuple[float, float]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    The bar the Sharpe must clear rises with the number of variants tried and how much
    their Sharpes differ. Returns (DSR, the monthly Sharpe bar used).
    """
    n = len(trial_sharpes_monthly)
    if n <= 1:
        bar = 0.0
    else:
        sd = float(np.std(trial_sharpes_monthly, ddof=1))
        bar = sd * ((1 - EULER_GAMMA) * N.inv_cdf(1 - 1 / n) + EULER_GAMMA * N.inv_cdf(1 - 1 / (n * math.e)))
    return probabilistic_sharpe(excess, bar), bar


def summary(r: pd.Series, b: pd.Series, rf: pd.Series) -> dict:
    return {"cagr": cagr(r), "vol": volatility(r), "sharpe": sharpe(r, rf), "max_dd": max_drawdown(r),
            "bench_cagr": cagr(b), "bench_vol": volatility(b), "bench_sharpe": sharpe(b, rf),
            "bench_max_dd": max_drawdown(b), "excess": cagr(r) - cagr(b), "te": tracking_error(r, b),
            "ir": information_ratio(r, b), "beta": beta(r, b), "hit": hit_rate(r, b),
            "rel_dd": relative_drawdown(r, b)}
