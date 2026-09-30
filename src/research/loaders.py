"""
Cached warehouse access for the research stack.

The problem this fixes
----------------------
A single `athlete-deep-dive` run used to issue roughly fourteen full
warehouse pulls: `load_pitching_trials_wide()` unfiltered for the athlete
context, again per stratum for the session-change section, again for the
outlier cohort, and then once more *inside* `correlate_kinematic_to_assessments`
for every flagged metric — each of those also re-pulling the entire profile
matrix. Same bytes, same process, over and over.

The fix
-------
Pull the full trial table and the full profile matrix once per process, then
derive every stratum by filtering in pandas. Age-group filtering in SQL and
age-group filtering in a DataFrame give the same rows; only one of them costs
a round trip.

    from src.research.loaders import trials, profiles, clear_cache

    all_trials = trials()                    # one query, cached
    pro        = trials(age_group="PRO")     # filtered from the cache
    prof       = profiles(role="pitcher")    # one query, cached

Cached frames are returned as copies by default, so a caller that mutates
(and several do) cannot corrupt the cache. Pass `copy=False` in hot loops
where you know you are read-only.

Scope: process lifetime. A CLI invocation is the unit. Long-lived callers
should call `clear_cache()` when they want fresh data — there is deliberately
no TTL, because a silently-refreshing cache would make two sections of the
same report disagree about the data.
"""
from __future__ import annotations

import threading
from typing import Any

import pandas as pd

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}
_STATS = {"hits": 0, "misses": 0, "queries": 0}


def clear_cache() -> None:
    """Drop everything. Call between logical runs in a long-lived process."""
    with _LOCK:
        _CACHE.clear()
        _STATS.update(hits=0, misses=0, queries=0)


def cache_stats() -> dict[str, int]:
    return dict(_STATS)


def _memo(key: str, producer):
    with _LOCK:
        if key in _CACHE:
            _STATS["hits"] += 1
            return _CACHE[key]
    # Produce outside the lock — a warehouse query can take seconds and we do
    # not want to serialise unrelated callers behind it.
    value = producer()
    with _LOCK:
        _STATS["misses"] += 1
        _CACHE.setdefault(key, value)
        return _CACHE[key]


def _out(df: pd.DataFrame, copy: bool) -> pd.DataFrame:
    return df.copy() if copy else df


# ──────────────────────────────────────────────────────────────────────────
# Trials
# ──────────────────────────────────────────────────────────────────────────

def trials(
    *,
    age_group: str | None = None,
    min_velocity: float | None = None,
    max_velocity: float | None = None,
    min_as_of_date: str | None = None,
    include_force_metrics: bool = True,
    copy: bool = True,
) -> pd.DataFrame:
    """Trial-level pitching data. One warehouse pull per (include_force_metrics,
    min_as_of_date) combination; every other filter is applied in pandas."""
    base_key = f"trials::fm={include_force_metrics}::since={min_as_of_date}"

    def _load():
        from src.research.pitching_deep import load_pitching_trials_wide
        _STATS["queries"] += 1
        return load_pitching_trials_wide(
            include_force_metrics=include_force_metrics,
            min_as_of_date=min_as_of_date,
        )

    df = _memo(base_key, _load)
    if df.empty:
        return df.copy() if copy else df

    out = df
    if age_group:
        want = str(age_group).strip().upper()
        ag = out["age_group"].astype("string").str.strip().str.upper()
        out = out[ag == want]
    if min_velocity is not None:
        out = out[out["velocity_mph"] >= min_velocity]
    if max_velocity is not None:
        out = out[out["velocity_mph"] <= max_velocity]
    if out is df:
        return _out(df, copy)
    return out.reset_index(drop=True)


def trials_for_athlete(athlete_uuid: str, *, copy: bool = True) -> pd.DataFrame:
    df = trials(copy=False)
    if df.empty:
        return df.copy() if copy else df
    return df[df["athlete_uuid"] == athlete_uuid].reset_index(drop=True)


def athlete_counts_by_stratum() -> pd.Series:
    """Athletes per age group — used to decide stratum vs global before any
    expensive work starts."""
    df = trials(copy=False)
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("age_group")["athlete_uuid"].nunique().sort_values(ascending=False)


# ──────────────────────────────────────────────────────────────────────────
# Profiles
# ──────────────────────────────────────────────────────────────────────────

def profiles(
    *,
    role: str | None = None,
    age_group: str | None = None,
    latest_only: bool = True,
    min_as_of_date: str | None = None,
    exclude_mobility: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """Athlete × metric profile matrix. Cached on the arguments that change
    the SQL; role and age_group are applied in pandas."""
    base_key = (f"profiles::latest={latest_only}::since={min_as_of_date}"
                f"::nomob={exclude_mobility}")

    def _load():
        from src.research.profile_matrix import load_matrix
        _STATS["queries"] += 1
        return load_matrix(
            role=None, age_group=None, latest_only=latest_only,
            min_as_of_date=min_as_of_date, exclude_mobility=exclude_mobility,
        )

    df = _memo(base_key, _load)
    if df.empty:
        return df.copy() if copy else df

    out = df
    if age_group and "age_group" in out.columns:
        want = str(age_group).strip().upper()
        ag = out["age_group"].astype("string").str.strip().str.upper()
        out = out[ag == want]
    if role and role != "all" and "role" in out.columns:
        out = out[out["role"] == "both"] if role == "both" \
            else out[out["role"].isin([role, "both"])]
    if out is df:
        return _out(df, copy)
    return out.reset_index(drop=True)


def profiles_all_dates(*, copy: bool = True) -> pd.DataFrame:
    """Every profile row for every athlete (serial, not latest-only)."""
    def _load():
        from src.research.session_change import load_profiles_all_dates
        _STATS["queries"] += 1
        return load_profiles_all_dates()
    return _out(_memo("profiles_all_dates", _load), copy)


# ──────────────────────────────────────────────────────────────────────────
# Stratum resolution — one rule, used everywhere
# ──────────────────────────────────────────────────────────────────────────

class StratumChoice:
    """The outcome of 'should I analyse this athlete against his own age
    group, or fall back to everyone?' — recorded so the report can say which
    lens it used and why, instead of silently switching."""

    def __init__(self, requested: str | None, used: str,
                 n_athletes: int, fell_back: bool, note: str | None):
        self.requested = requested
        self.used = used
        self.n_athletes = n_athletes
        self.fell_back = fell_back
        self.note = note

    GLOBAL = "ALL ATHLETES"

    def __repr__(self) -> str:
        return (f"StratumChoice(requested={self.requested!r}, used={self.used!r}, "
                f"n={self.n_athletes}, fell_back={self.fell_back})")

    def to_dict(self) -> dict[str, Any]:
        return {"requested": self.requested, "used": self.used,
                "n_athletes": self.n_athletes, "fell_back": self.fell_back,
                "note": self.note}


def resolve_stratum(age_group: str | None, *, min_athletes: int
                    ) -> tuple[pd.DataFrame, StratumChoice]:
    """Return (trial_frame, choice) for the tightest usable comparison group."""
    if age_group:
        sub = trials(age_group=age_group, copy=False)
        n = int(sub["athlete_uuid"].nunique()) if not sub.empty else 0
        if n >= min_athletes:
            return sub, StratumChoice(age_group, age_group, n, False, None)
        allt = trials(copy=False)
        n_all = int(allt["athlete_uuid"].nunique()) if not allt.empty else 0
        note = (f"Only {n} athlete(s) on file in {age_group} — needs "
                f"{min_athletes} to be its own comparison group, so this "
                f"section compares against all {n_all} pitchers instead. "
                f"Read the percentiles as 'vs everyone we test', not "
                f"'vs his level'.")
        return allt, StratumChoice(age_group, StratumChoice.GLOBAL, n_all, True, note)

    allt = trials(copy=False)
    n_all = int(allt["athlete_uuid"].nunique()) if not allt.empty else 0
    return allt, StratumChoice(None, StratumChoice.GLOBAL, n_all, False,
                               "No age group on file for this athlete.")


# ──────────────────────────────────────────────────────────────────────────
# Reliability
# ──────────────────────────────────────────────────────────────────────────

def reliability_table(*, age_group: str | None = None):
    """The measurement-noise floor for every trial metric, computed once.

    Deliberately estimated on the WHOLE cohort rather than per stratum: the
    scatter of a repeated measurement is a property of the capture system and
    the metric, not of how hard the athlete throws, and pooling gives an
    estimate stable enough to trust. `age_group` is accepted for callers that
    genuinely want a stratum-specific floor and is cached separately.
    """
    from src.research.reliability import (
        ReliabilityTable, compute_reliability, estimate_between_session,
        merge_reliability,
    )
    key = f"reliability::{age_group or 'ALL'}"

    def _build():
        from src.research.pitching_deep import metric_columns_pitching
        df = trials(age_group=age_group, copy=False)
        if df.empty:
            return ReliabilityTable()
        metrics = metric_columns_pitching(
            df, processed_only=True, exclude_symptomatic=False, role="pitcher",
        ) + (["velocity_mph"] if "velocity_mph" in df.columns else [])
        within = compute_reliability(df, metrics)
        between = estimate_between_session(df, metrics)
        return ReliabilityTable(merge_reliability(within, between))

    return _memo(key, _build)


__all__ = [
    "trials", "trials_for_athlete", "profiles", "profiles_all_dates",
    "athlete_counts_by_stratum", "resolve_stratum", "StratumChoice",
    "reliability_table", "clear_cache", "cache_stats",
]
