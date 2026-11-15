"""Choosing the stocks to hold each month (Design Doc 7.1 and 2.4)."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def index_sector_weights(cands: pd.DataFrame) -> pd.Series:
    """Approximate S&P 500 sector weights from the universe's market values."""
    mc = cands["market_cap"].where(cands["market_cap"] > 0)
    w = mc.groupby(cands["sector"]).sum()
    return w / w.sum() if w.sum() > 0 else w


def sector_bounds(index_w: pd.Series, n: int, band: float = 0.05) -> tuple[dict, dict]:
    """Minimum and maximum number of names per sector for index weight +/- `band`."""
    mins = {s: max(0, math.ceil((w - band) * n - 1e-9)) for s, w in index_w.items()}
    maxs = {s: max(0, math.floor((w + band) * n + 1e-9)) for s, w in index_w.items()}
    return mins, maxs


def select(cands: pd.DataFrame, held: set, n: int, buffer: float, mins: dict, maxs: dict,
           must_keep: set | frozenset = frozenset()) -> list[str]:
    """Pick `n` stocks.

    0. Keep every stock in `must_keep` (holdings in their waiting period).
    1. Meet each sector's minimum with its best-ranked names (current holdings inside
       the buffer first).
    2. Keep current holdings ranked within buffer x n (the buffer rule).
    3. Add the best-ranked new names.
    No sector goes above its maximum unless there is no other way to reach `n`.
    """
    by_rank = cands.sort_values("rank").reset_index(drop=True)
    chosen, count = [], defaultdict(int)
    chosen_set = set()

    def add(t, s):
        chosen.append(t)
        chosen_set.add(t)
        count[s] += 1

    for t, s in zip(by_rank["ticker"], by_rank["sector"]):
        if t in must_keep and t not in chosen_set:
            add(t, s)

    keep_zone = by_rank["rank"] <= buffer * n
    for s, m in sorted(mins.items(), key=lambda kv: -kv[1]):
        pool = by_rank[by_rank["sector"] == s]
        first = pool[pool["ticker"].isin(held) & keep_zone[pool.index]]
        for t in pd.concat([first, pool.drop(first.index)])["ticker"]:
            if count[s] >= m or len(chosen) >= n:
                break
            add(t, s)

    for t, s, r in zip(by_rank["ticker"], by_rank["sector"], by_rank["rank"]):
        if len(chosen) >= n:
            break
        if t in held and r <= buffer * n and t not in chosen_set and count[s] < maxs.get(s, n):
            add(t, s)

    for t, s in zip(by_rank["ticker"], by_rank["sector"]):
        if len(chosen) >= n:
            break
        if t not in chosen_set and count[s] < maxs.get(s, n):
            add(t, s)

    for t, s in zip(by_rank["ticker"], by_rank["sector"]):   # only if sector caps made n impossible
        if len(chosen) >= n:
            break
        if t not in chosen_set:
            add(t, s)
    return chosen
