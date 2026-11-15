"""Shared helpers for turning raw characteristics into comparable scores (Design Doc, Section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Section 6.2: weights fixed in advance from the 3/7 risk profile, not optimised on data.
WEIGHTS = {"low_vol": 0.30, "quality": 0.30, "res_mom": 0.25, "value": 0.15}
MIN_SIGNALS = 3  # Section 6.3: a stock needs at least 3 of the 4 scores


def rank_score(values: pd.Series, groups: pd.Series | None = None) -> pd.Series:
    """Scale values to [-1, +1] by rank: the highest value gets +1, the lowest -1.

    With `groups` (e.g. sector), stocks are ranked only against their own group.
    Missing values stay missing. A group with one stock scores 0. Because only the
    order matters, extreme values cannot dominate (no separate winsorising needed).
    """
    def one(x: pd.Series) -> pd.Series:
        n = x.notna().sum()
        if n <= 1:
            return x * 0.0
        return 2.0 * (x.rank(method="average") - 1.0) / (n - 1.0) - 1.0

    if groups is None:
        return one(values)
    g = groups.fillna("Unknown")
    return values.groupby(g, group_keys=False).transform(one)


def mean_of_available(parts: list[pd.Series], min_parts: int) -> pd.Series:
    """Average of the parts that exist; missing when fewer than `min_parts` exist."""
    df = pd.concat(parts, axis=1)
    return df.mean(axis=1, skipna=True).where(df.notna().sum(axis=1) >= min_parts)


def combine(scores: pd.DataFrame, weights: dict[str, float] = WEIGHTS, min_signals: int = MIN_SIGNALS) -> pd.Series:
    """Weighted composite of the four scores.

    A missing score counts as 0, the sector median (Section 6.3). Stocks with fewer
    than `min_signals` scores get no composite.
    """
    cols = list(weights)
    available = scores[cols].notna().sum(axis=1)
    total = sum(scores[c].fillna(0.0) * w for c, w in weights.items())
    return total.where(available >= min_signals)


def spearman_by_date(df: pd.DataFrame, score: str, outcome: str) -> pd.Series:
    """Rank correlation between a score and an outcome, one value per date (the IC)."""
    def ic(g: pd.DataFrame) -> float:
        g = g[[score, outcome]].dropna()
        if len(g) < 20:
            return np.nan
        return g[score].rank().corr(g[outcome].rank())
    return df.groupby("date")[[score, outcome]].apply(ic)
