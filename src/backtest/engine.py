"""Month-by-month simulation of the strategy with contributions and trading costs.

Everything is tracked in USD (the stocks' currency); EUR results are derived from the
ECB rate on each trade date. Costs follow Design Doc 9.2.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.backtest.portfolio import index_sector_weights, sector_bounds, select
from src.signals.common import WEIGHTS, combine


@dataclass(frozen=True)
class Config:
    label: str = "Base case"
    start_capital_eur: float = 10_000.0
    contribution_eur: float = 1_500.0
    n_small: int = 25                    # holdings while capital < n_switch_eur (Section 1.1)
    n_large: int = 40
    n_switch_eur: float = 25_000.0
    fixed_n: int | None = None           # robustness: force a holding count
    buffer: float = 1.5                  # keep a holding while ranked within 1.5 x N
    sector_band: float = 0.05            # sector weight within index weight +/- 5 points
    resize_band: tuple = (0.6, 1.6)      # resize only outside 60%-160% of target weight
    cash_target: float = 0.02
    min_trade_eur: float = 50.0
    commission_min_usd: float = 1.0      # Section 9.2: max(USD 1, 0.05%) per order
    commission_rate: float = 0.0005
    spread_slippage_bps: float = 20.0    # mid/small caps: half-spread + slippage (Section 9.2, DMF-US-02)
    fx_bps: float = 1.0                  # converting each contribution
    cost_multiplier: float = 1.0         # robustness: 2x costs
    trade_day: int = 2                   # trade on the 2nd trading day of the month
    weights: tuple = tuple(WEIGHTS.items())
    min_signals: int = 3
    # Trading rules carried over from the S&P 500 project's trial 2 (Design Doc 7.1, 7.2 and 2.4).
    trim_rule: str = "limit"             # "limit": trim only above the single-stock limit + drift
                                         # "band": resize to target outside resize_band (S&P 500 trial 1)
    exit_after_months: int = 2           # sell after this many consecutive month-ends outside buffer x N
    max_weight_small: float = 0.05       # single-stock limit with n_small holdings
    max_weight_large: float = 0.04       # ... with more holdings
    drift_pp: float = 0.01               # trim only when the limit is exceeded by this much

    def key(self) -> str:
        """Fingerprint of the settings (the label is ignored)."""
        d = asdict(self)
        d.pop("label")
        return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]


def rerank(scores: pd.DataFrame, weights: dict, min_signals: int) -> pd.DataFrame:
    """Recompute the composite and rank with different weights (robustness tests)."""
    if weights == WEIGHTS and min_signals == 3:
        return scores
    s = scores.copy()
    s["composite"] = combine(s, weights, min_signals)
    s["rank"] = s.groupby("date")["composite"].rank(ascending=False, method="first")
    return s


def simulate(scores: pd.DataFrame, windows: pd.DataFrame, periods: dict, fx_start: pd.Series, fx_end: pd.Series,
             cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the strategy. Returns (monthly results, trades)."""
    scores = rerank(scores, dict(cfg.weights), cfg.min_signals)
    by_date = {d: g for d, g in scores[scores["composite"].notna()].groupby("date")}
    holdings: dict[str, float] = {}
    outside: dict[str, int] = {}          # consecutive month-ends a holding has ranked outside buffer x N
    waiting_stats = {"recovered": 0, "sold_after_wait": 0}
    cash = 0.0
    rows, trades = [], []
    m = cfg.cost_multiplier
    first = True

    for F in windows.index:
        if F not in by_date or F not in periods:
            continue
        ret = periods[F]
        start, end = windows.loc[F, "start"], windows.loc[F, "end"]
        fx0, fx1 = fx_start[F], fx_end[F]

        for t in [t for t in holdings if t not in ret.index]:   # stopped trading: already cashed out
            cash += holdings.pop(t)
        value_before = sum(holdings.values()) + cash
        contrib_usd = (cfg.start_capital_eur if first else cfg.contribution_eur) * fx0
        first = False
        cash += contrib_usd
        total = value_before + contrib_usd

        n = cfg.fixed_n or (cfg.n_small if value_before / fx0 < cfg.n_switch_eur else cfg.n_large)
        cands = by_date[F][by_date[F]["ticker"].isin(ret.index)]
        mins, maxs = sector_bounds(index_sector_weights(cands), n, cfg.sector_band)

        # Waiting period: a holding that ranks outside buffer x N is kept until it has done so
        # at `exit_after_months` consecutive month-ends. Leaving the universe still means selling.
        rank_of = dict(zip(cands["ticker"], cands["rank"]))
        waiting = set()
        for t in holdings:
            r = rank_of.get(t)
            if r is None:
                continue
            if r > cfg.buffer * n:
                outside[t] = outside.get(t, 0) + 1
                if outside[t] < cfg.exit_after_months:
                    waiting.add(t)
            else:
                if outside.get(t, 0) > 0:
                    waiting_stats["recovered"] += 1
                outside[t] = 0
        chosen = select(cands, set(holdings), n, cfg.buffer, mins, maxs, must_keep=waiting)
        target = total * (1 - cfg.cash_target) / n
        lo, hi = cfg.resize_band
        max_w = cfg.max_weight_small if n <= cfg.n_small else cfg.max_weight_large
        min_trade = cfg.min_trade_eur * fx0

        orders = {t: -v for t, v in holdings.items() if t not in chosen}          # sells
        for t in orders:
            if cfg.exit_after_months > 1 and outside.get(t, 0) >= cfg.exit_after_months:
                waiting_stats["sold_after_wait"] += 1
            outside.pop(t, None)
        for t in chosen:
            v = holdings.get(t, 0.0)
            if v == 0.0:
                orders[t] = target                                                # new name
            elif cfg.trim_rule == "band":
                if (v < lo * target and t not in waiting) or v > hi * target:
                    orders[t] = target - v                                        # resize (trial 1)
            elif v > (max_w + cfg.drift_pp) * total:
                orders[t] = max_w * total - v                                     # trim back to the limit
            elif v < lo * target and t not in waiting:
                orders[t] = target - v                                            # top up a small position
        sells = -sum(a for a in orders.values() if a < 0)
        buys = sum(a for a in orders.values() if a > 0)
        budget = cash + sells - total * cfg.cash_target
        if buys > budget:
            scale = max(budget, 0.0) / buys if buys > 0 else 0.0
            orders = {t: (a * scale if a > 0 else a) for t, a in orders.items()}
            buys *= scale
        leftover = budget - buys
        if leftover > min_trade:                                                  # contributions -> underweights
            gaps = sorted(((target - holdings[t], t) for t in chosen if t in holdings and t not in orders
                           and t not in waiting and holdings[t] < target), reverse=True)
            for gap, t in gaps:
                if leftover <= min_trade:
                    break
                amt = min(gap, leftover)
                orders[t] = amt
                leftover -= amt
        orders = {t: a for t, a in orders.items() if abs(a) >= min_trade or (a < 0 and t not in chosen)}

        cost = cfg.fx_bps / 1e4 * contrib_usd * m
        for t, a in orders.items():
            c = (max(cfg.commission_min_usd, cfg.commission_rate * abs(a)) + cfg.spread_slippage_bps / 1e4 * abs(a)) * m
            cost += c
            holdings[t] = holdings.get(t, 0.0) + a
            cash -= a
            trades.append({"formation": F, "trade_date": start, "ticker": t, "usd": a, "cost_usd": c})
        cash -= cost
        holdings = {t: v for t, v in holdings.items() if v > 1e-6}

        for t in holdings:
            holdings[t] *= 1.0 + ret.get(t, 0.0)
        value_after = sum(holdings.values()) + cash
        r_usd = value_after / total - 1.0
        rows.append({
            "month": (F + pd.offsets.MonthEnd(1)).normalize(), "formation": F, "start": start, "end": end,
            "ret_usd": r_usd, "ret_eur": (1 + r_usd) * fx0 / fx1 - 1, "value_eur": value_after / fx1,
            "contribution_eur": contrib_usd / fx0, "holdings": len(holdings), "n_target": n,
            "orders": len(orders), "cost_eur": cost / fx0, "sells_eur": sum(-a for a in orders.values() if a < 0) / fx0,
            "buys_eur": sum(a for a in orders.values() if a > 0) / fx0, "cash_share": cash / value_after,
            "waiting": len(waiting),
        })
    out = pd.DataFrame(rows).set_index("month")
    out.attrs["waiting_stats"] = waiting_stats
    return out, pd.DataFrame(trades)
