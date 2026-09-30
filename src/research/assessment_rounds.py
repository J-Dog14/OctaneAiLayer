"""
Assessment-round grouping — 3D-anchored ±14-day windows.

The rule: a pitching-3D or hitting-3D capture is the TRUTH for when an
athlete was assessed. Any mobility / athletic screen / proteus / arm-strength
result within ±window_days of that anchor is part of that round. Anything
outside every anchor's window is excluded from the deep-dive entirely.

Downstream analyses (session-change, big-picture) use rounds as the unit of
comparison — first round vs latest round — instead of the raw profile
snapshot dates.

Design notes:
  - We only need per-source *dates* to bucket. Values still come from
    ai_layer.athlete_profiles (where the profiler already computed Z-scores),
    filtered by the round window.
  - Rounds are named "R1", "R2", ... in chronological anchor order.
  - Anchor priority: pitching_3d > hitting_3d (a pitcher can hit; pitching
    is our primary lens). If an athlete only has hitting-3D anchors, those
    are used instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from src.db import backend_conn, query


DEFAULT_WINDOW_DAYS = 14

# Every table we consider a "source" for a round. Each has (athlete_uuid,
# session_date) as its temporal key.
_SOURCE_TABLES: dict[str, str] = {
    "mobility":              "public.f_mobility",
    "screen_dj":             "public.f_athletic_screen_dj",
    "screen_cmj":            "public.f_athletic_screen_cmj",
    "screen_ppu":            "public.f_athletic_screen_ppu",
    "screen_slv":            "public.f_athletic_screen_slv",
    # NMT is discontinued (last taken 2024). Leaving it in made every report
    # carry a coverage row and an unprofiled-date warning for a test nobody
    # will take again.
    # "screen_nmt":          "public.f_athletic_screen_nmt",
    "proteus":               "public.f_proteus",
    "arm_action":            "public.f_arm_action",
    "curveball_test":        "public.f_curveball_test",
    "arm_care":              "public.f_pro_sup",  # ProSup / arm-care
}

# Anchors: pitching-3D preferred; hitting-3D fallback.
_ANCHOR_PITCHING = "public.f_pitching_trials"
_ANCHOR_HITTING = "public.f_hitting_trials"


@dataclass
class AssessmentRound:
    round_id: str                      # 'R1', 'R2', ...
    anchor_type: str                   # 'pitching_3d' | 'hitting_3d'
    anchor_date: Any                   # date
    window_start: Any
    window_end: Any
    sources_present: dict[str, Any]    # source_name → actual date within window
    profile_id: int | None = None      # ai_layer.athlete_profiles row inside window
    profile_as_of_date: Any = None


def _list_athlete_dates(conn, table: str, uuid: str) -> list[date]:
    sql = f"SELECT DISTINCT session_date FROM {table} " \
          f"WHERE athlete_uuid = %s ORDER BY session_date"
    rows = query(conn, sql, [uuid])
    return [r["session_date"] for r in rows]


def _list_profile_rows(conn, uuid: str) -> list[dict]:
    sql = """
        SELECT id, as_of_date
        FROM ai_layer.athlete_profiles
        WHERE athlete_uuid = %s
        ORDER BY as_of_date
    """
    return list(query(conn, sql, [uuid]))


def build_assessment_rounds(
    athlete_uuid: str,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    prefer_hitting: bool = False,
) -> tuple[list[AssessmentRound], dict[str, Any]]:
    """Return (rounds, meta).

    meta contains:
      - anchor_type used ('pitching_3d' | 'hitting_3d' | None)
      - orphaned assessments: source → [dates outside every window]
      - window_days
    """
    with backend_conn() as conn:
        pitching_anchors = _list_athlete_dates(conn, _ANCHOR_PITCHING, athlete_uuid)
        hitting_anchors = _list_athlete_dates(conn, _ANCHOR_HITTING, athlete_uuid)
        anchor_type = None
        anchors: list[date] = []
        if prefer_hitting and hitting_anchors:
            anchor_type = "hitting_3d"
            anchors = hitting_anchors
        elif pitching_anchors:
            anchor_type = "pitching_3d"
            anchors = pitching_anchors
        elif hitting_anchors:
            anchor_type = "hitting_3d"
            anchors = hitting_anchors
        else:
            return [], {"anchor_type": None,
                        "orphaned_assessments": {},
                        "window_days": window_days,
                        "note": "No 3D anchors on record — no rounds can be built."}

        source_dates: dict[str, list[date]] = {}
        for name, table in _SOURCE_TABLES.items():
            try:
                source_dates[name] = _list_athlete_dates(conn, table, athlete_uuid)
            except Exception:
                source_dates[name] = []
        profile_rows = _list_profile_rows(conn, athlete_uuid)

    # Build rounds
    rounds: list[AssessmentRound] = []
    win = timedelta(days=window_days)

    # An anchor "session cluster": if the athlete threw two 3D sessions within
    # window_days of each other, treat them as one round. (Rare but real —
    # e.g. two days in a row of testing.)
    clustered: list[list[date]] = []
    for a in anchors:
        if clustered and (a - clustered[-1][-1]) <= win:
            clustered[-1].append(a)
        else:
            clustered.append([a])

    for i, cluster in enumerate(clustered, start=1):
        anchor_date = cluster[-1]  # use latest date of the cluster as the anchor
        window_start = min(cluster) - win
        window_end = max(cluster) + win

        # Match sources into this window; keep the source date closest to anchor
        present: dict[str, Any] = {}
        for name, dates in source_dates.items():
            candidates = [d for d in dates
                          if window_start <= d <= window_end]
            if candidates:
                present[name] = min(candidates,
                                     key=lambda d: abs((d - anchor_date).days))

        # Match profile inside window
        matched_profile_id = None
        matched_profile_date = None
        best_gap = None
        for pr in profile_rows:
            pd_ = pr["as_of_date"]
            if window_start <= pd_ <= window_end:
                gap = abs((pd_ - anchor_date).days)
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    matched_profile_id = pr["id"]
                    matched_profile_date = pd_

        rounds.append(AssessmentRound(
            round_id=f"R{i}",
            anchor_type=anchor_type,
            anchor_date=anchor_date,
            window_start=window_start,
            window_end=window_end,
            sources_present=present,
            profile_id=matched_profile_id,
            profile_as_of_date=matched_profile_date,
        ))

    # Orphaned assessments — dates >window away from every anchor
    orphaned: dict[str, list[date]] = {}
    for name, dates in source_dates.items():
        oo = []
        for d in dates:
            in_any = False
            for r in rounds:
                if r.window_start <= d <= r.window_end:
                    in_any = True
                    break
            if not in_any:
                oo.append(d)
        if oo:
            orphaned[name] = oo

    # Every date per source, not just the one nearest each anchor. Without
    # this, coverage undercounts badly: a round keeps ONE date per source (the
    # closest to the anchor), so an athlete who does Proteus twice a week has
    # every other session inside the window counted nowhere — neither paired
    # nor orphaned. It reported 11 Proteus dates for an athlete with 41.
    all_dates = {name: list(dates) for name, dates in source_dates.items()
                 if dates}
    in_window: dict[str, list[date]] = {}
    for name, dates in source_dates.items():
        hits = [d for d in dates
                if any(r.window_start <= d <= r.window_end for r in rounds)]
        if hits:
            in_window[name] = hits

    meta = {
        "anchor_type": anchor_type,
        "n_anchors": len(anchors),
        "n_rounds": len(rounds),
        "orphaned_assessments": orphaned,
        "all_source_dates": all_dates,
        "dates_in_a_window": in_window,
        "window_days": window_days,
    }
    return rounds, meta


def rounds_to_dataframe(rounds: list[AssessmentRound]) -> pd.DataFrame:
    """Flat table summarizing rounds for display."""
    rows = []
    for r in rounds:
        row: dict[str, Any] = {
            "round_id": r.round_id,
            "anchor_type": r.anchor_type,
            "anchor_date": r.anchor_date,
            "window_start": r.window_start,
            "window_end": r.window_end,
            "profile_as_of_date": r.profile_as_of_date,
            "n_sources": len(r.sources_present),
        }
        for name in _SOURCE_TABLES.keys():
            row[name] = r.sources_present.get(name)
        rows.append(row)
    return pd.DataFrame(rows)
