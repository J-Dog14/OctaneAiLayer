"""
Per-athlete deep-research package.

Composes the other research modules into one per-athlete analysis answering:

  1) Fingerprint    — what data do we have, and what does he look like now?
  2) Variability    — inside the focus session, which mechanics varied
                      pitch-to-pitch, and did the harder pitches look
                      mechanically different? (Only when there are enough
                      pitches to ask the question at all.)
  3) Round change   — round-to-round deltas in mechanics and in assessments,
                      each judged against that metric's measurement-error band.
  4) Standouts      — where he sits against the tightest usable comparison
                      group, as a percentile when the cohort supports one and
                      an honest rank when it does not.
  5) Velo drivers   — the within-athlete kin↔velo correlates for his stratum,
                      with his own value on each.

  + headlines       — the handful of findings that survived every filter,
                      written as sentences. This is what the coach page shows;
                      everything above is the appendix behind it.

What changed in v2 and why
--------------------------
* Every within-session correlation now has a hard n floor and a bootstrap
  interval, and the section reports what a top |r| would look like under pure
  noise for that n and that many metrics. At n=4 across 150 metrics the noise
  ceiling is |r| = 1.0, which is why the old table led with r = -0.80, p = 0.20
  and called it the top velocity correlate.
* Assessment deltas come from RAW values, not Z-scores. Z-scores are computed
  against a norms table that is rebuilt in place with no versioning, so a
  Z-delta silently mixed real change with norm drift.
* An assessment only enters a round's delta if its own source session actually
  falls inside that round's window. The profiler carries the last value
  forward indefinitely, so a round with no mobility test still had mobility
  Z-scores — differencing those against a fresh test invented change.
* Cohort percentiles are one row per ATHLETE, not per athlete-session. A
  pitcher with six sessions used to contribute six rows and bend the scale.
* Every delta carries a verdict against that metric's minimal detectable
  change, so "moved 6 degrees" becomes "moved 6 degrees against a 2.1 degree
  error band".
* The warehouse is read once per process (see `loaders`), not fourteen times.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from src.db import backend_conn, query
from src.metrics_spec import METRICS
from src.research import loaders
from src.research.assessment_rounds import (
    AssessmentRound,
    build_assessment_rounds,
    rounds_to_dataframe,
)
from src.research.config import (
    CONFIG,
    TIER_INSUFFICIENT,
    TIER_STRONG,
    TIER_SUGGESTIVE,
    ResearchConfig,
)
from src.research.kinematic_drivers import correlate_kinematic_to_assessments
from src.research.metric_display import DISPLAY
from src.research.metric_filters import (
    FAMILY_LABEL,
    FAMILY_ORDER,
    KINEMATICS,
    KINETICS,
    classify_family,
    filter_for_change,
)
from src.research.session_matrix import (
    SessionMatrix,
    build_assessment_history,
    build_assessment_matrix,
    build_pitch_matrix,
    build_session_matrix,
    describe_change_window,
)
from src.research.unit_guard import audit_scale_changes
from src.research.pitching_deep import (
    correlate_velocity_within_athlete,
    metric_columns_pitching,
)
from src.research.profile_matrix import _classify_metric
from src.research.reliability import (
    VERDICT_BEYOND,
    VERDICT_EDGE,
    VERDICT_LIKELY,
    VERDICT_NOISE,
    VERDICT_NO_BASELINE,
    VERDICT_REAL,
    VERDICT_TYPICAL,
    VERDICT_UNKNOWN,
    ChangeVerdict,
    ReliabilityTable,
)
from src.research.stats_support import (
    CohortPosition,
    correlation_tier,
    max_abs_r_null,
    percentile_or_rank,
    spearman_with_ci,
)

# metric key → modality, so we can check whether a profile value was actually
# measured inside a round window or merely carried forward by the profiler.
METRIC_MODALITY: dict[str, str] = {m["key"]: m["modality"] for m in METRICS}

_PROFILE_META = {"profile_id", "as_of_date", "age_group"}


# ──────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    session_date: Any
    n_trials: int
    mean_velocity: float | None
    max_velocity: float | None


@dataclass
class ProfileSnapshot:
    """One ai_layer.athlete_profiles row, with the provenance needed to decide
    whether each value is fresh or a carry-forward."""
    profile_id: int
    as_of_date: Any
    age_group: str | None
    z_scores: dict[str, float]
    raw_values: dict[str, float]
    source_dates: dict[str, Any]

    def source_date_for(self, metric: str) -> Any | None:
        mod = METRIC_MODALITY.get(metric)
        if mod is None:
            return None
        v = self.source_dates.get(mod)
        return _as_date(v)


@dataclass
class AthleteContext:
    uuid: str
    name: str
    email: str | None
    age_group: str | None
    handedness: str | None
    height: float | None
    weight: float | None
    trial_df: pd.DataFrame
    profile_df: pd.DataFrame
    latest_profile: pd.Series | None
    sessions: list[SessionInfo]
    focus_session_date: Any
    rounds: list[AssessmentRound] = field(default_factory=list)
    rounds_meta: dict = field(default_factory=dict)
    snapshots: dict[int, ProfileSnapshot] = field(default_factory=dict)
    reliability: ReliabilityTable = field(default_factory=ReliabilityTable)
    unit_audit: Any = None          # UnitAudit — scale breaks between sessions

    def trials_on(self, session_date: Any) -> pd.DataFrame:
        if self.trial_df.empty:
            return self.trial_df
        return self.trial_df[self.trial_df["session_date"] == session_date]

    def n_trials_on(self, session_date: Any) -> int:
        return int(len(self.trials_on(session_date)))


@dataclass
class Finding:
    """One thing worth saying out loud, already written as a sentence.

    The coach page is built entirely from these; it never reaches into the
    section dictionaries. That keeps 'what we decided to tell the coach' in
    one auditable place instead of spread across a renderer.
    """
    kind: str                 # change | standout | limiter | variability | load
    metric: str
    headline: str
    detail: str
    tier: str
    verdict: str = VERDICT_UNKNOWN
    good: int | None = None   # +1 toward better, -1 away, None = no direction
    value: float | None = None
    delta: float | None = None
    mdc: float | None = None
    percentile: float | None = None
    rank: int | None = None
    cohort_n: int = 0
    cohort_label: str = ""
    group: str = "other"
    cue: str | None = None
    provisional: bool = False   # band came from a stand-in, not test-retest
    priority: float = 0.0     # higher sorts first

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["display_name"] = DISPLAY.name(self.metric)
        return d


@dataclass
class DeepDiveReport:
    athlete: AthleteContext
    fingerprint: dict
    variability: dict
    session_change: dict
    outliers: dict
    velo_drivers: dict
    big_picture: dict = field(default_factory=dict)
    assessment_coverage: dict = field(default_factory=dict)
    headlines: list[Finding] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: ResearchConfig = CONFIG
    generated_at: datetime = field(default_factory=datetime.now)

    # convenience for renderers / persistence
    def findings_frame(self) -> pd.DataFrame:
        if not self.headlines:
            return pd.DataFrame()
        return pd.DataFrame([f.to_dict() for f in self.headlines])


def _as_date(v: Any) -> Any | None:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return pd.to_datetime(v).date()
    except Exception:
        return None


def _days_between(a: Any, b: Any) -> int | None:
    da, db = _as_date(a), _as_date(b)
    if da is None or db is None:
        return None
    return abs((db - da).days)


# ──────────────────────────────────────────────────────────────────────────
# Athlete resolution + context loading
# ──────────────────────────────────────────────────────────────────────────

def resolve_athlete(query_str: str) -> tuple[str, str, str | None]:
    """Resolve email / uuid / name to (uuid, name, email)."""
    is_uuid = (len(query_str) == 36 and query_str.count("-") == 4)
    sql_email = """SELECT athlete_uuid, name, email FROM analytics.d_athletes
                   WHERE LOWER(email) = LOWER(%s) LIMIT 1"""
    sql_uuid = """SELECT athlete_uuid, name, email FROM analytics.d_athletes
                  WHERE athlete_uuid = %s LIMIT 1"""
    sql_name_exact = """SELECT athlete_uuid, name, email FROM analytics.d_athletes
                        WHERE LOWER(name) = LOWER(%s) LIMIT 1"""
    sql_name_like = """SELECT athlete_uuid, name, email FROM analytics.d_athletes
                       WHERE name ILIKE %s ORDER BY name LIMIT 5"""
    with backend_conn() as conn:
        if is_uuid:
            rows = query(conn, sql_uuid, [query_str])
            if rows:
                r = rows[0]
                return r["athlete_uuid"], r["name"], r.get("email")
        rows = query(conn, sql_email, [query_str])
        if rows:
            r = rows[0]
            return r["athlete_uuid"], r["name"], r.get("email")

        # Exact name. Duplicate athlete records are common — the warehouse has
        # a whole merge script for them — and the old `LIMIT 1` here silently
        # picked one, which is how a freshly ingested session on a second
        # athlete_uuid looks like "the report is showing stale data". Take the
        # record that actually has the most recent capture, and say so.
        rows = query(conn, """
            SELECT d.athlete_uuid, d.name, d.email,
                   (SELECT MAX(session_date) FROM public.f_pitching_trials t
                     WHERE t.athlete_uuid = d.athlete_uuid) AS last_pitching
            FROM analytics.d_athletes d
            WHERE LOWER(d.name) = LOWER(%s)
        """, [query_str])
        if rows:
            if len(rows) > 1:
                best = max(rows, key=lambda r: (r["last_pitching"] or date.min))
                others = [r for r in rows if r is not best]
                print(
                    f"[resolve_athlete] WARNING: {len(rows)} athlete records "
                    f"share the name {query_str!r}. Using the one with the most "
                    f"recent pitching capture "
                    f"({best['last_pitching']}, {best['athlete_uuid']}); the "
                    f"others are "
                    + ", ".join(f"{r['athlete_uuid']} (last capture "
                                f"{r['last_pitching']})" for r in others)
                    + ". These are probably duplicates that need merging — run "
                      "`research diagnose` for the full picture."
                )
                return best["athlete_uuid"], best["name"], best.get("email")
            r = rows[0]
            return r["athlete_uuid"], r["name"], r.get("email")

        rows = query(conn, sql_name_like, [f"%{query_str}%"])
        if len(rows) == 1:
            r = rows[0]
            return r["athlete_uuid"], r["name"], r.get("email")
        if len(rows) > 1:
            names = ", ".join(r["name"] for r in rows)
            raise ValueError(
                f"{query_str!r} matched multiple athletes: {names}. "
                f"Use a full name, email, or athlete_uuid."
            )
    raise ValueError(f"No athlete matched {query_str!r}.")


def _load_profile_snapshots(uuid: str) -> tuple[pd.DataFrame, dict[int, ProfileSnapshot]]:
    """Every profile row for one athlete — Z-scores as a wide frame (for
    backwards compatibility) plus the full snapshot with raw values and
    source dates, which is what the change analysis actually uses."""
    sql = """
        SELECT p.id, p.as_of_date, p.age_group,
               p.z_scores, p.raw_values, p.source_dates
        FROM ai_layer.athlete_profiles p
        WHERE p.athlete_uuid = %s
        ORDER BY p.as_of_date
    """
    with backend_conn() as conn:
        rows = query(conn, sql, [uuid])
    if not rows:
        return pd.DataFrame(), {}

    records: list[dict[str, Any]] = []
    snaps: dict[int, ProfileSnapshot] = {}
    for r in rows:
        z = r["z_scores"] or {}
        raw = r["raw_values"] or {}
        rec: dict[str, Any] = {
            "profile_id": r["id"],
            "as_of_date": r["as_of_date"],
            "age_group": r["age_group"],
        }
        zf: dict[str, float] = {}
        for k, v in z.items():
            f = _coerce(v)
            rec[k] = f
            if f is not None:
                zf[k] = f
        snaps[r["id"]] = ProfileSnapshot(
            profile_id=r["id"],
            as_of_date=r["as_of_date"],
            age_group=r["age_group"],
            z_scores=zf,
            raw_values={k: f for k, v in raw.items()
                        if (f := _coerce(v)) is not None},
            source_dates=dict(r["source_dates"] or {}),
        )
        records.append(rec)
    return pd.DataFrame(records), snaps


def _coerce(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def load_athlete_context(uuid: str, *, focus_session_date: Any = None,
                         config: ResearchConfig = CONFIG) -> AthleteContext:
    """Everything one athlete's deep dive needs. One warehouse read per table
    per process, courtesy of `loaders`."""
    trial_df = loaders.trials_for_athlete(uuid)

    if not trial_df.empty:
        name = trial_df["name"].iloc[0]
        age_group = trial_df["age_group"].iloc[0]
        handedness = trial_df["handedness"].iloc[0]
        height = trial_df["height"].iloc[0]
        weight = trial_df["weight"].iloc[0]
    else:
        name, age_group, handedness, height, weight = "(unknown)", None, None, None, None

    with backend_conn() as conn:
        rows = query(conn, """
            SELECT name, email FROM analytics.d_athletes WHERE athlete_uuid = %s
        """, [uuid])
    email = rows[0].get("email") if rows else None
    if name == "(unknown)" and rows:
        name = rows[0]["name"]

    sessions: list[SessionInfo] = []
    if not trial_df.empty:
        for sd, g in trial_df.groupby("session_date"):
            v = g["velocity_mph"].dropna()
            sessions.append(SessionInfo(
                session_date=sd, n_trials=len(g),
                mean_velocity=float(v.mean()) if len(v) else None,
                max_velocity=float(v.max()) if len(v) else None,
            ))
        sessions.sort(key=lambda s: s.session_date)

    if focus_session_date is None and sessions:
        focus_session_date = sessions[-1].session_date

    profile_df, snapshots = _load_profile_snapshots(uuid)
    latest_profile = None
    if not profile_df.empty:
        latest_profile = profile_df.sort_values("as_of_date").iloc[-1]
    if age_group is None and latest_profile is not None:
        age_group = latest_profile.get("age_group")

    try:
        rounds, rounds_meta = build_assessment_rounds(
            uuid, window_days=config.round_window_days)
    except Exception as e:  # a missing source table must not kill the report
        rounds, rounds_meta = [], {"error": str(e)}

    try:
        reliability = loaders.reliability_table()
    except Exception:
        reliability = ReliabilityTable()

    # Scale/unit discontinuities between this athlete's own sessions. Found in
    # real data: every GRF metric was ~1000x smaller after a pipeline change,
    # which the change table happily reported as a 2,106 N loss of drive-leg
    # force. Detect it here so the comparison can decline instead.
    unit_audit = None
    if not trial_df.empty:
        try:
            audit_metrics = metric_columns_pitching(
                trial_df, processed_only=True, exclude_symptomatic=False,
                role="pitcher")
            unit_audit = audit_scale_changes(trial_df, audit_metrics)
        except Exception:
            unit_audit = None

    return AthleteContext(
        uuid=uuid, name=name, email=email, age_group=age_group,
        handedness=handedness, height=height, weight=weight,
        trial_df=trial_df, profile_df=profile_df, latest_profile=latest_profile,
        sessions=sessions, focus_session_date=focus_session_date,
        rounds=rounds, rounds_meta=rounds_meta, snapshots=snapshots,
        reliability=reliability, unit_audit=unit_audit,
    )


# ──────────────────────────────────────────────────────────────────────────
# Section 1: Fingerprint
# ──────────────────────────────────────────────────────────────────────────

def section_fingerprint(ctx: AthleteContext) -> dict:
    populated: list[dict] = []
    if ctx.latest_profile is not None:
        for k, v in ctx.latest_profile.items():
            if k in _PROFILE_META or v is None:
                continue
            vf = _coerce(v)
            if vf is None:
                continue
            populated.append({
                "metric": k,
                "display_name": DISPLAY.name(k),
                "domain": _classify_metric(k),
                "z_score": round(vf, 3),
            })
    z_df = (pd.DataFrame(populated).sort_values(["domain", "metric"])
              .reset_index(drop=True)) if populated else pd.DataFrame()

    by_domain: dict[str, int] = {}
    for r in populated:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1

    max_velo = latest_velo = latest_n = None
    if ctx.sessions:
        max_velo = max((s.max_velocity for s in ctx.sessions
                        if s.max_velocity is not None), default=None)
        latest = ctx.sessions[-1]
        latest_velo = latest.mean_velocity
        latest_n = latest.n_trials

    span_days = None
    if len(ctx.sessions) >= 2:
        span_days = _days_between(ctx.sessions[0].session_date,
                                  ctx.sessions[-1].session_date)

    return {
        "n_profiles": len(ctx.profile_df),
        "n_pitching_sessions": len(ctx.sessions),
        "n_pitching_trials": int(len(ctx.trial_df)),
        "history_span_days": span_days,
        "domain_coverage": by_domain,
        "populated_zscores": z_df,
        "max_velocity_career": max_velo,
        "latest_session_mean_velocity": latest_velo,
        "latest_session_n_trials": latest_n,
        "sessions": pd.DataFrame([{
            "session_date": s.session_date, "n_trials": s.n_trials,
            "mean_velocity": s.mean_velocity, "max_velocity": s.max_velocity,
        } for s in ctx.sessions]) if ctx.sessions else pd.DataFrame(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Section 2: Trial-to-trial variability (focus session)
# ──────────────────────────────────────────────────────────────────────────

def section_trial_variability(ctx: AthleteContext, *,
                              config: ResearchConfig = CONFIG,
                              top_k: int = 20) -> dict:
    """Pitch by pitch, inside the focus session.

    This ALWAYS runs when there are two or more pitches. Every value is a
    measurement of something that happened, and the spread between pitches is
    itself the interesting thing — a pitcher who moves eight degrees of
    hip-shoulder separation from pitch to pitch is a different pitcher from one
    who moves two, and that is a finding, not an error term. With exactly two
    pitches the honest statement is the difference between them, so that is
    what gets reported.

    One thing is still guarded, and it is a guard about a PROCEDURE rather than
    about the data: ranking ~150 metrics by correlation with velocity across a
    handful of pitches and reading the top of the list. The ranking itself
    manufactures a large value — at four pitches across 150 metrics the biggest
    |r| you get from unrelated columns is 1.00 — so that table is labelled as a
    scan and its scan-wide reference value is stated, rather than the top row
    being promoted to a finding. The per-metric values are all shown either way.
    """
    if ctx.trial_df.empty or ctx.focus_session_date is None:
        return {"error": "No trial data on record."}

    focus = ctx.trials_on(ctx.focus_session_date).copy()
    n_focus = len(focus)
    if n_focus < 2:
        return {"error": f"The focus session has {n_focus} pitch — nothing to "
                         f"compare it against within the session."}

    metrics = metric_columns_pitching(focus, processed_only=True,
                                      exclude_symptomatic=True, role="pitcher")
    metrics, _dropped = filter_for_change(metrics)
    if not metrics:
        return {"error": "No pitching metrics present in the focus session."}

    min_n_for_corr = config.min_trials_within_session
    hist = ctx.trial_df[ctx.trial_df["session_date"] != ctx.focus_session_date]

    rows: list[dict] = []
    for m in metrics:
        vals = focus[m].dropna()
        if len(vals) < 2:
            continue
        mean_v = float(vals.mean())
        sd_v = float(vals.std()) if len(vals) > 1 else 0.0
        lo, hi = float(vals.min()), float(vals.max())
        rng = hi - lo
        cv = float(sd_v / abs(mean_v)) if mean_v != 0 else None

        # With two pitches the difference IS the story; with more, the range.
        pair_diff = None
        if len(vals) == 2:
            ordered = focus.sort_values("trial_index")[m].dropna()
            pair_diff = float(ordered.iloc[1] - ordered.iloc[0])

        res = (spearman_with_ci(focus[m], focus["velocity_mph"], config=config)
               if len(vals) >= 3 else None)

        hist_cv = hist_ratio = None
        if not hist.empty:
            hv = hist[m].dropna()
            if len(hv) >= config.min_hist_trials_for_cv and float(hv.mean()) != 0:
                hist_cv = float(hv.std() / abs(hv.mean()))
                if cv is not None and hist_cv > 0:
                    hist_ratio = float(cv / hist_cv)

        rel = ctx.reliability.get(m)
        rows.append({
            "metric": m,
            "display_name": DISPLAY.name(m),
            "family": classify_family(m),
            "unit": DISPLAY.unit(m),
            "n_pitches": int(len(vals)),
            "mean": round(mean_v, 3),
            "sd": round(sd_v, 3),
            "min": round(lo, 3),
            "max": round(hi, 3),
            "range": round(rng, 3),
            "pitch_to_pitch_diff": round(pair_diff, 3) if pair_diff is not None else None,
            "cv": round(cv, 3) if cv is not None else None,
            "hist_cv": round(hist_cv, 3) if hist_cv is not None else None,
            "cv_vs_hist_ratio": round(hist_ratio, 2) if hist_ratio is not None else None,
            "usual_spread": round(rel.sem, 3) if rel else None,
            "r_vs_velo": round(res.r, 3) if res and res.r is not None else None,
            "ci_low": round(res.ci_low, 3) if res and res.ci_low is not None else None,
            "ci_high": round(res.ci_high, 3) if res and res.ci_high is not None else None,
            "ci": res.format_ci() if res else "—",
            "p_vs_velo": round(res.p_value, 4) if res and res.p_value is not None else None,
            "n_trials": res.n if res else int(len(vals)),
            "tier": (correlation_tier(res, min_n=min_n_for_corr, config=config)
                     if res else TIER_INSUFFICIENT),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "No metric had two or more valid pitches."}

    # ── What moved most between pitches, per family ───────────────────────
    df["spread_vs_usual"] = [
        (round(r["range"] / r["usual_spread"], 2)
         if r.get("usual_spread") else None)
        for _, r in df.iterrows()
    ]
    df["spread_vs_usual"] = pd.to_numeric(df["spread_vs_usual"], errors="coerce")
    by_family: dict[str, pd.DataFrame] = {}
    for fam, sub in df.groupby("family"):
        by_family[fam] = (sub.sort_values("spread_vs_usual", ascending=False,
                                          na_position="last")
                             .head(top_k).reset_index(drop=True))

    # ── The per-pitch table for the metrics that moved most ───────────────
    movers = (df.dropna(subset=["spread_vs_usual"])
                .sort_values("spread_vs_usual", ascending=False)
                .head(8)["metric"].tolist()) or df.head(8)["metric"].tolist()
    per_pitch = focus.sort_values("trial_index")[
        ["trial_index", "velocity_mph"] + [m for m in movers if m in focus.columns]
    ].reset_index(drop=True)

    # ── The scan, clearly labelled as a scan ──────────────────────────────
    # With two pitches every correlation is None, which leaves an object-dtype
    # column that .abs() cannot handle — coerce first.
    df["r_vs_velo"] = pd.to_numeric(df["r_vs_velo"], errors="coerce")
    df["abs_r"] = df["r_vs_velo"].abs()
    ranked = df.dropna(subset=["r_vs_velo"]).sort_values("abs_r", ascending=False)
    null = None
    credible = pd.DataFrame()
    if not ranked.empty:
        null = max_abs_r_null(
            n=int(ranked["n_trials"].median()),
            n_metrics=int(len(ranked)),
            observed_max_abs_r=float(ranked["abs_r"].iloc[0]),
            n_sim=600,
        )
        credible = ranked[
            (ranked["ci_low"].notna())
            & (ranked["ci_low"] * ranked["ci_high"] > 0)
            & (ranked["abs_r"] >= null.null_p95)
        ].drop(columns=["abs_r"]).reset_index(drop=True)

    top_velo_correlates = (ranked.head(top_k).drop(columns=["abs_r"])
                                 .reset_index(drop=True)
                           if not ranked.empty else pd.DataFrame())

    destabilizing = (df.dropna(subset=["cv_vs_hist_ratio"])
                       .query("cv_vs_hist_ratio >= @config.destabilizing_cv_ratio")
                       .sort_values("cv_vs_hist_ratio", ascending=False)
                       .head(top_k).drop(columns=["abs_r"], errors="ignore")
                       .reset_index(drop=True))

    velo = focus["velocity_mph"]
    velo_spread = None
    if velo.notna().sum() >= 2:
        velo_spread = {
            "mean": round(float(velo.mean()), 2),
            "sd": round(float(velo.std()), 2),
            "min": round(float(velo.min()), 1),
            "max": round(float(velo.max()), 1),
            "range": round(float(velo.max() - velo.min()), 1),
        }

    # The same table shape as the capture matrix, with pitches as the columns
    # instead of dates — so the session reads without learning a new layout.
    pitch_order = (df.dropna(subset=["spread_vs_usual"])
                     .sort_values("spread_vs_usual", ascending=False)["metric"]
                     .tolist()) or df["metric"].tolist()
    # Velocity is the column a coach reads first, so it belongs in the table
    # rather than only in the sentence above it.
    if "velocity_mph" in focus.columns and "velocity_mph" not in pitch_order:
        pitch_order = ["velocity_mph"] + pitch_order
    pitch_matrix = build_pitch_matrix(focus, pitch_order, ctx.reliability,
                                      config=config)

    return {
        "focus_session_date": ctx.focus_session_date,
        "n_trials": n_focus,
        "pitch_matrix": pitch_matrix,
        "velo_mean": velo_spread["mean"] if velo_spread else None,
        "velo_std": velo_spread["sd"] if velo_spread else None,
        "velo_spread": velo_spread,
        "n_metrics_scanned": int(len(ranked)),
        "correlation_floor": min_n_for_corr,
        "correlations_underpowered": bool(n_focus < min_n_for_corr),
        "noise_ceiling": null,
        "by_family": by_family,
        "per_pitch": per_pitch,
        "movers": movers,
        "top_velo_correlates": top_velo_correlates,
        "credible_correlates": credible,
        "destabilizing_metrics": destabilizing,
        "all_metrics": df.drop(columns=["abs_r"], errors="ignore").reset_index(drop=True),
    }


# ──────────────────────────────────────────────────────────────────────────
# Section 3: Round-to-round change
# ──────────────────────────────────────────────────────────────────────────

def _session_mean(ctx: AthleteContext, session_date: Any,
                  metrics: list[str]) -> tuple[pd.Series | None, int]:
    tf = ctx.trials_on(session_date)
    if tf.empty:
        return None, 0
    cols = [c for c in metrics if c in tf.columns]
    if "velocity_mph" in tf.columns and "velocity_mph" not in cols:
        cols.append("velocity_mph")   # appending it twice selects it twice
    return tf[cols].mean(), int(len(tf))


def _snapshot_for_round(ctx: AthleteContext, rnd: AssessmentRound
                        ) -> ProfileSnapshot | None:
    if rnd.profile_id is None:
        return None
    return ctx.snapshots.get(rnd.profile_id)


def _assessment_deltas_between_rounds(
    ctx: AthleteContext, r_from: AssessmentRound, r_to: AssessmentRound,
    *, config: ResearchConfig = CONFIG,
) -> tuple[pd.DataFrame, dict]:
    """Raw-value assessment deltas, gated on provenance.

    Two rules, both of which the Z-score version silently broke:

      1. Difference RAW values, not Z-scores. Z-scores are computed against
         `ai_layer.assessment_norms`, which is rebuilt in place with no
         version stamp. Two profiles built months apart were scored against
         different norm tables, so part of every Z-delta was the population
         moving, not the athlete.

      2. A metric only counts if its own source session sits inside BOTH
         round windows. `profiler` takes the latest session at-or-before the
         as-of date with no recency limit, so a round whose window contains no
         mobility test still carries mobility values from whenever they were
         last measured. Differencing a carry-forward against a fresh test
         manufactures change out of nothing.

    Everything excluded is returned in the meta dict so the report can say
    what it dropped and why, rather than quietly showing a shorter table.
    """
    snap_from = _snapshot_for_round(ctx, r_from)
    snap_to = _snapshot_for_round(ctx, r_to)
    meta: dict[str, Any] = {
        "from_profile_id": snap_from.profile_id if snap_from else None,
        "to_profile_id": snap_to.profile_id if snap_to else None,
        "excluded_stale": [],
        "excluded_same_source": [],
        "n_compared": 0,
    }
    if snap_from is None or snap_to is None:
        meta["error"] = "One or both rounds have no matched profile snapshot."
        return pd.DataFrame(), meta

    rows: list[dict] = []
    shared = set(snap_from.raw_values) & set(snap_to.raw_values)
    for k in sorted(shared):
        v1 = snap_from.raw_values.get(k)
        v2 = snap_to.raw_values.get(k)
        if v1 is None or v2 is None:
            continue

        d_from = snap_from.source_date_for(k)
        d_to = snap_to.source_date_for(k)

        # Rule 2a — the same measurement on both ends is not a change.
        if d_from is not None and d_to is not None and d_from == d_to:
            meta["excluded_same_source"].append(
                {"metric": k, "source_date": str(d_from)})
            continue

        # Rule 2b — a value measured far outside its round window is a
        # carry-forward, not an observation belonging to that round.
        stale = []
        for label, snap, d, rnd in (("from", snap_from, d_from, r_from),
                                    ("to", snap_to, d_to, r_to)):
            if d is None:
                continue
            gap = _days_between(d, rnd.anchor_date)
            if gap is not None and gap > config.max_source_staleness_days:
                stale.append({"end": label, "source_date": str(d),
                              "anchor": str(rnd.anchor_date), "gap_days": gap})
        if stale:
            meta["excluded_stale"].append({"metric": k, "reasons": stale})
            continue

        verb, formatted = DISPLAY.describe_change(k, float(v2) - float(v1))
        rows.append({
            "metric": k,
            "display_name": DISPLAY.name(k),
            "domain": _classify_metric(k),
            "group": DISPLAY.group(k),
            "unit": DISPLAY.unit(k),
            "from_value": round(float(v1), 3),
            "to_value": round(float(v2), 3),
            "delta": round(float(v2) - float(v1), 3),
            "pct_change": (round(100.0 * (float(v2) - float(v1)) / abs(float(v1)), 1)
                           if float(v1) != 0 else None),
            "from_source_date": str(d_from) if d_from else None,
            "to_source_date": str(d_to) if d_to else None,
            "change": verb,
            "change_text": formatted,
            "toward_better": DISPLAY.signed_toward_better(k, float(v2) - float(v1)),
            "z_from": snap_from.z_scores.get(k),
            "z_to": snap_to.z_scores.get(k),
        })

    meta["n_compared"] = len(rows)
    if not rows:
        return pd.DataFrame(), meta
    out = pd.DataFrame(rows)
    out["abs_pct"] = out["pct_change"].abs()
    return (out.sort_values("abs_pct", ascending=False, na_position="last")
               .drop(columns=["abs_pct"]).reset_index(drop=True)), meta


def _mechanic_deltas_between(
    ctx: AthleteContext, date_from: Any, date_to: Any, metrics: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Session-mean mechanic deltas, split by family and provenance-checked.

    Three filters, all of which earned their place on real data:

      * model constants are excluded outright. Segment lengths and frame rate
        are fixed by the marker set and the rig; they change when a tech places
        a marker differently, not when the pitcher changes. They are also
        near-constant within a session, so their measured spread is ~0, and
        dividing by ~0 sorted them to the very top of the table with ratios of
        7.79e+14.
      * bodyweight is kept once, in kg, not twice in kg and newtons.
      * anything that crossed a unit/scale break between these two dates is
        reported as a unit change rather than differenced.

    Returns (frame, meta) — meta carries what was dropped and why, because a
    silently shorter table is how the segment-length problem survived a whole
    report cycle.
    """
    meta: dict[str, Any] = {"excluded_model": {}, "excluded_scale_break": {}}

    metrics, dropped = filter_for_change(metrics)
    meta["excluded_model"] = dropped

    blocked: set[str] = set()
    if ctx.unit_audit is not None:
        try:
            blocked = ctx.unit_audit.blocked_metrics_between(date_from, date_to)
        except Exception:
            blocked = set()
    if blocked:
        meta["excluded_scale_break"] = {
            m: "units or scale changed between these two captures"
            for m in blocked if m in metrics}
        metrics = [m for m in metrics if m not in blocked]

    m_from, n_from = _session_mean(ctx, date_from, metrics)
    m_to, n_to = _session_mean(ctx, date_to, metrics)
    if m_from is None or m_to is None:
        return pd.DataFrame(), meta

    rows: list[dict] = []
    for m in metrics:
        v1, v2 = m_from.get(m), m_to.get(m)
        if pd.isna(v1) or pd.isna(v2):
            continue
        delta = float(v2) - float(v1)
        cv: ChangeVerdict = ctx.reliability.classify(
            m, delta, n_before=n_from, n_after=n_to)
        verb, formatted = DISPLAY.describe_change(m, delta)
        rows.append({
            "metric": m,
            "display_name": DISPLAY.name(m),
            "family": classify_family(m),
            "group": DISPLAY.group(m),
            "unit": DISPLAY.unit(m),
            "from_value": round(float(v1), 3),
            "to_value": round(float(v2), 3),
            "delta": round(delta, 3),
            "pct_change": (round(100.0 * delta / abs(float(v1)), 1)
                           if float(v1) != 0 else None),
            "change": verb,
            "change_text": formatted,
            "toward_better": DISPLAY.signed_toward_better(m, delta),
            "usual_spread": round(cv.mdc, 3) if cv.mdc is not None else None,
            "mdc": round(cv.mdc, 3) if cv.mdc is not None else None,
            "spread_multiple": round(cv.ratio, 2) if cv.ratio is not None else None,
            "mdc_ratio": round(cv.ratio, 2) if cv.ratio is not None else None,
            "verdict": cv.verdict,
            "is_real": cv.exceeds_typical,
            "provisional": cv.provisional,
            "n_before": n_from,
            "n_after": n_to,
        })
    if not rows:
        return pd.DataFrame(), meta
    out = pd.DataFrame(rows)
    out["_sort"] = out["spread_multiple"].fillna(-1.0)
    out = (out.sort_values("_sort", ascending=False)
              .drop(columns=["_sort"]).reset_index(drop=True))
    return out, meta


def split_deltas_by_family(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Kinetics and kinematics get their own tables.

    There are far more force metrics in the warehouse than joint-angle ones, so
    a single table sorted by magnitude buries the kinematics behind the
    kinetics every time — and it takes both to see what actually happened.
    """
    if df is None or df.empty:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam]
        if not sub.empty:
            out[fam] = sub.reset_index(drop=True)
    for fam, sub in df.groupby("family"):
        out.setdefault(fam, sub.reset_index(drop=True))
    return out


def section_session_change(ctx: AthleteContext, *,
                           config: ResearchConfig = CONFIG,
                           top_k_kin_correlates: int = 15) -> dict:
    """Round 1 → latest round, in mechanics and in assessments."""
    if len(ctx.rounds) < 2:
        if len(ctx.sessions) < 2:
            return {"skipped": "Only one pitching session on record — "
                               "nothing to compare against yet."}
        return {"skipped": (
            f"Only {len(ctx.rounds)} assessment round could be built. Change "
            f"analysis needs two 3D captures with assessments inside their "
            f"±{config.round_window_days}-day windows.")}

    first_r, latest_r = ctx.rounds[0], ctx.rounds[-1]
    prev_r = ctx.rounds[-2]

    metrics = metric_columns_pitching(ctx.trial_df, processed_only=True,
                                      exclude_symptomatic=True, role="pitcher")
    if not metrics:
        return {"error": "No pitching metrics available."}
    # Velocity is the outcome every other row is trying to explain, so it
    # belongs in the table with them. Leaving it out also broke the headline:
    # with no velocity row to read, the finding fell back to the first-to-last
    # delta while keeping the latest-vs-previous label, and reported "down 1.9
    # mph since 2025-10-24" for an athlete who went 91.9 -> 87.5 between those
    # two captures. The real move was 4.4.
    if "velocity_mph" in ctx.trial_df.columns and "velocity_mph" not in metrics:
        metrics = ["velocity_mph"] + list(metrics)

    # The metrics x sessions matrix. Latest-vs-previous is the headline
    # comparison; the earlier captures stay on screen as columns so a move can
    # be read against its own history rather than against a point 735 days ago.
    matrix = build_session_matrix(ctx, metrics, config=config)

    mech_first, mech_meta = _mechanic_deltas_between(
        ctx, first_r.anchor_date, latest_r.anchor_date, metrics)
    if prev_r is not latest_r:
        mech_prev, _ = _mechanic_deltas_between(
            ctx, prev_r.anchor_date, latest_r.anchor_date, metrics)
    else:
        mech_prev = pd.DataFrame()

    m_first, n_first = _session_mean(ctx, first_r.anchor_date, metrics)
    m_latest, n_latest = _session_mean(ctx, latest_r.anchor_date, metrics)
    m_prev, n_prev = _session_mean(ctx, prev_r.anchor_date, metrics)
    if m_first is None or m_latest is None:
        return {"error": "No trials on one of the round anchor dates."}

    def _dv(a, b, na, nb):
        if a is None or b is None:
            return None, None
        va, vb = a.get("velocity_mph"), b.get("velocity_mph")
        if pd.isna(va) or pd.isna(vb):
            return None, None
        d = float(vb) - float(va)
        return d, ctx.reliability.classify("velocity_mph", d,
                                           n_before=na, n_after=nb)

    dv_first, dv_first_verdict = _dv(m_first, m_latest, n_first, n_latest)
    dv_prev, dv_prev_verdict = _dv(m_prev, m_latest, n_prev, n_latest)

    span_days = _days_between(first_r.anchor_date, latest_r.anchor_date)
    span_prev = _days_between(prev_r.anchor_date, latest_r.anchor_date)

    asmt_first, asmt_meta = _assessment_deltas_between_rounds(
        ctx, first_r, latest_r, config=config)

    # ── Stratum velo drivers joined against this athlete's movement ───────
    strat_trials, choice = loaders.resolve_stratum(
        ctx.age_group, min_athletes=config.min_athletes_for_stratum)
    velo_correlate_join = pd.DataFrame()
    if not strat_trials.empty:
        try:
            within = correlate_velocity_within_athlete(
                strat_trials,
                min_trials_per_athlete=config.min_trials_per_athlete,
                min_n=config.min_n_within_athlete,
                fdr_alpha=config.fdr_alpha,
                processed_only=True, exclude_symptomatic=True,
            )
        except Exception:
            within = pd.DataFrame()
        if not within.empty and not mech_first.empty:
            by_metric = mech_first.set_index("metric")
            joined: list[dict] = []
            for _, row in within.head(top_k_kin_correlates).iterrows():
                m = row["metric"]
                if m not in by_metric.index:
                    continue
                d = by_metric.loc[m]
                delta = d.get("delta")
                if pd.isna(delta):
                    continue
                stratum_r = float(row["r"])
                aligned = bool(np.sign(delta) == np.sign(stratum_r))
                # "Moved the right way" only counts when the move was bigger
                # than the noise. Signing a delta of 0.2 against an error band
                # of 3.0 is coin-flipping with extra steps.
                meaningful = bool(d.get("is_real"))
                joined.append({
                    "metric": m,
                    "display_name": DISPLAY.name(m),
                    "stratum_r_vs_velo": round(stratum_r, 3),
                    "stratum_fdr_sig": bool(row.get("fdr_significant", False)),
                    "athlete_delta": round(float(delta), 3),
                    "mdc": d.get("mdc"),
                    "verdict": d.get("verdict"),
                    "moved_right_way": aligned,
                    "counts": aligned and meaningful,
                })
            velo_correlate_join = pd.DataFrame(joined)

    n_aligned = int(velo_correlate_join["counts"].sum()) if not velo_correlate_join.empty else 0
    n_joined = int(len(velo_correlate_join))

    return {
        "round1_id": first_r.round_id,
        "roundN_id": latest_r.round_id,
        "prev_round_id": prev_r.round_id if prev_r is not latest_r else None,
        "round1_anchor_date": first_r.anchor_date,
        "roundN_anchor_date": latest_r.anchor_date,
        "prev_round_anchor_date": prev_r.anchor_date if prev_r is not latest_r else None,
        "round1_sources": first_r.sources_present,
        "roundN_sources": latest_r.sources_present,
        "span_days": span_days,
        "span_days_prev": span_prev,
        "long_span": bool(span_days and span_days > config.long_span_warn_days),
        "n_trials_first": n_first,
        "n_trials_latest": n_latest,
        "delta_velocity_first": round(dv_first, 3) if dv_first is not None else None,
        "delta_velocity_first_verdict": dv_first_verdict,
        "delta_velocity_prev": round(dv_prev, 3) if dv_prev is not None else None,
        "delta_velocity_prev_verdict": dv_prev_verdict,
        "matrix": matrix,
        "matrix_window": describe_change_window(matrix),
        "assessment_matrix": build_assessment_matrix(ctx, config=config),
        "assessment_history": build_assessment_history(ctx, config=config),
        "mechanic_deltas": mech_first,
        "mechanic_deltas_by_family": split_deltas_by_family(mech_first),
        "mechanic_delta_meta": mech_meta,
        "mechanic_deltas_prev": mech_prev,
        "assessment_deltas": asmt_first,
        "assessment_delta_meta": asmt_meta,
        "velo_correlate_join": velo_correlate_join,
        "n_aligned": n_aligned,
        "n_joined": n_joined,
        "stratum": choice.to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Section 3.5: Big picture — every round transition
# ──────────────────────────────────────────────────────────────────────────

def section_big_picture(ctx: AthleteContext, *,
                        config: ResearchConfig = CONFIG,
                        top_k_each_side: int = 10) -> dict:
    if len(ctx.rounds) < 2:
        return {"skipped": "Needs at least two assessment rounds."}

    metrics = metric_columns_pitching(ctx.trial_df, processed_only=True,
                                      exclude_symptomatic=True, role="pitcher")
    transitions: list[dict] = []
    for i in range(1, len(ctx.rounds)):
        r_prev, r_curr = ctx.rounds[i - 1], ctx.rounds[i]
        mech, mech_meta = _mechanic_deltas_between(
            ctx, r_prev.anchor_date, r_curr.anchor_date, metrics)
        asmt, asmt_meta = _assessment_deltas_between_rounds(
            ctx, r_prev, r_curr, config=config)

        m_prev, n_prev = _session_mean(ctx, r_prev.anchor_date, metrics)
        m_curr, n_curr = _session_mean(ctx, r_curr.anchor_date, metrics)
        dv = dv_verdict = None
        if m_prev is not None and m_curr is not None:
            a, b = m_prev.get("velocity_mph"), m_curr.get("velocity_mph")
            if pd.notna(a) and pd.notna(b):
                dv = float(b) - float(a)
                dv_verdict = ctx.reliability.classify(
                    "velocity_mph", dv, n_before=n_prev, n_after=n_curr)

        span = _days_between(r_prev.anchor_date, r_curr.anchor_date)
        real_mech = mech[mech["is_real"]] if not mech.empty else pd.DataFrame()

        transitions.append({
            "from_round": r_prev.round_id,
            "to_round": r_curr.round_id,
            "from_date": r_prev.anchor_date,
            "to_date": r_curr.anchor_date,
            "span_days": span,
            "long_span": bool(span and span > config.long_span_warn_days),
            "delta_velocity": round(dv, 3) if dv is not None else None,
            "delta_velocity_verdict": dv_verdict,
            "n_real_mechanic_changes": int(len(real_mech)),
            "mechanic_delta_meta": mech_meta,
            "mechanic_deltas_by_family": split_deltas_by_family(mech),
            "top_mechanic_changes": (real_mech.head(top_k_each_side)
                                     if not real_mech.empty
                                     else mech.head(top_k_each_side)),
            "showing_only_real": bool(len(real_mech) > 0),
            "top_assessment_changes": asmt.head(top_k_each_side) if not asmt.empty
                                      else pd.DataFrame(),
            "assessment_meta": asmt_meta,
        })
    return {"n_transitions": len(transitions), "transitions": transitions}


# ──────────────────────────────────────────────────────────────────────────
# Section 4: Standouts vs the cohort
# ──────────────────────────────────────────────────────────────────────────

def section_assessment_coverage(ctx: AthleteContext, *,
                                config: ResearchConfig = CONFIG) -> dict:
    """Everything he has been tested on, and whether it reached the analysis.

    Rounds pair assessments with a 3D capture, which is right for asking "what
    moved together". But the facility tests on its own cadence — screens
    quarterly, Proteus weekly, 3D twice a year — so plenty of real testing
    lands outside every 3D window. Dropping it silently made it look like the
    data was not there. This section lists it: what exists, what made it into a
    round, and what did not.
    """
    rounds = ctx.rounds or []
    meta = ctx.rounds_meta or {}
    window = meta.get("window_days", config.round_window_days)

    # Count every date on file, not just the one date per source that each
    # round keeps. A round records the assessment date CLOSEST to its anchor,
    # so an athlete on Proteus twice a week has most of his sessions inside a
    # window but not named by it — those used to be counted neither as paired
    # nor as outside, and simply vanished from the coverage total.
    all_dates: dict[str, list] = meta.get("all_source_dates", {}) or {}
    in_window: dict[str, list] = meta.get("dates_in_a_window", {}) or {}
    orphaned = meta.get("orphaned_assessments", {}) or {}

    anchor_dates: dict[str, list] = {}
    for r in rounds:
        for src, d in (r.sources_present or {}).items():
            anchor_dates.setdefault(src, []).append(d)

    if not all_dates:   # older rounds_meta — fall back to the narrow counts
        all_dates = {src: sorted(set(anchor_dates.get(src, []))
                                 | set(orphaned.get(src, [])))
                     for src in set(anchor_dates) | set(orphaned)}
        in_window = anchor_dates

    rows: list[dict] = []
    for src in sorted(all_dates):
        every = sorted(all_dates.get(src, []))
        paired = sorted(in_window.get(src, []))
        used = sorted(anchor_dates.get(src, []))
        rows.append({
            "source": src,
            "total_dates": len(every),
            "in_a_round": len(paired),
            "outside_every_round": len(every) - len(paired),
            "first": str(min(every)) if every else None,
            "latest": str(max(every)) if every else None,
            "latest_used": str(max(used)) if used else None,
        })
    df = pd.DataFrame(rows)
    n_missed = int(df["outside_every_round"].sum()) if not df.empty else 0
    n_total = int(df["total_dates"].sum()) if not df.empty else 0

    # Which test dates the profiler has never seen. A date that exists in the
    # fact tables but in no profile snapshot cannot appear anywhere in this
    # report, however wide the round window is — the values are read from
    # ai_layer.athlete_profiles. This is the difference between "not tested"
    # and "tested, but the profile table has not been rebuilt since".
    profiled: set = set()
    for snap in (ctx.snapshots or {}).values():
        for v in (snap.source_dates or {}).values():
            d = _as_date(v)
            if d is not None:
                profiled.add(d)
        if snap.as_of_date is not None:
            profiled.add(_as_date(snap.as_of_date))
    every_date: set = set()
    for dates in all_dates.values():
        every_date.update(dates)
    unprofiled = sorted(d for d in every_date if d not in profiled)

    return {
        "window_days": window,
        "n_rounds": len(rounds),
        "by_source": df,
        "n_profiles": len(ctx.snapshots or {}),
        "unprofiled_dates": [str(d) for d in unprofiled],
        "n_dates_total": n_total,
        "n_dates_outside": n_missed,
        "pct_used": (round(100.0 * (n_total - n_missed) / n_total, 0)
                     if n_total else None),
        "note": (
            f"A round is a 3D capture plus every assessment within "
            f"\u00b1{window} days of it, and rounds are used ONLY to link "
            f"assessments to mechanics. {n_total - n_missed} of {n_total} "
            f"assessment dates fell inside one — every date in the window, "
            f"not just the one closest to the capture. The rest are results "
            f"taken away from a capture \u2014 they are not missing, and they "
            f"are all in the assessment history above, which runs on his own "
            f"test dates rather than on the 3D calendar."
        ) if n_total else "No assessment dates on record.",
    }


def section_outliers(ctx: AthleteContext, *,
                     config: ResearchConfig = CONFIG,
                     run_drivers: bool = True) -> dict:
    """Where the athlete sits on each mechanic against the tightest usable
    comparison group.

    The cohort is one row per ATHLETE (mean of that athlete's session means),
    not one row per athlete-session. Under the old per-session aggregation a
    pitcher with six captures contributed six rows and pulled the scale toward
    himself, so a percentile was partly a statement about who happened to get
    tested most often.
    """
    if ctx.trial_df.empty or ctx.focus_session_date is None:
        return {"error": "No trial data on record."}
    focus = ctx.trials_on(ctx.focus_session_date)
    if focus.empty:
        return {"error": "Focus session has no trials."}

    metrics = metric_columns_pitching(focus, processed_only=True,
                                      exclude_symptomatic=True, role="pitcher")
    # The same exclusions the change tables use. Without them the cohort
    # section led with "Body weight: 100th percentile — higher than 23 of the
    # 23 PRO pitchers", twice, once in kilograms and once in newtons. He is
    # the biggest guy on file; that is not a finding about his delivery, and
    # printing it in two units is not two findings.
    metrics, _excluded_from_cohort = filter_for_change(metrics)
    if not metrics:
        return {"error": "No pitching metrics present."}

    cohort_trials, choice = loaders.resolve_stratum(
        ctx.age_group, min_athletes=config.min_athletes_for_stratum)
    if cohort_trials.empty:
        return {"error": "No cohort data available."}

    present = [m for m in metrics if m in cohort_trials.columns]
    # athlete-session means → athlete means. One row per athlete.
    per_session = (cohort_trials.groupby(["athlete_uuid", "session_date"])[present]
                                .mean().reset_index())
    per_athlete = per_session.groupby("athlete_uuid")[present].mean()

    athlete_mean = focus[present].mean()
    n_focus = int(len(focus))
    # Empty label reads as "pitchers we have tested" in the generated
    # sentence, which is the honest phrasing when we fell back to everyone.
    cohort_label = "" if choice.fell_back else (choice.used or "")

    rows: list[dict] = []
    for m in present:
        val = athlete_mean.get(m)
        if pd.isna(val):
            continue
        # Exclude the athlete himself so he is not compared against a cohort
        # that already contains him — with 9 athletes that self-inclusion
        # visibly shifts the percentile.
        cohort_vals = per_athlete[m].drop(index=ctx.uuid, errors="ignore").dropna()
        pos = percentile_or_rank(float(val), cohort_vals,
                                 config=config, cohort_label=cohort_label)
        if pos is None:
            continue
        rel = ctx.reliability.get(m)
        flag = None
        if pos.mode == "percentile" and pos.percentile is not None:
            if pos.percentile >= config.outlier_percentile:
                flag = "HIGH"
            elif pos.percentile <= (100 - config.outlier_percentile):
                flag = "LOW"
        elif pos.mode == "rank" and pos.rank is not None and pos.n >= 5:
            if pos.rank == 1:
                flag = "HIGH"
            elif pos.rank == pos.n:
                flag = "LOW"
        rows.append({
            "metric": m,
            "display_name": DISPLAY.name(m),
            "group": DISPLAY.group(m),
            "unit": DISPLAY.unit(m),
            "direction": DISPLAY.direction(m),
            "athlete_value": round(float(val), 3),
            "cohort_median": round(pos.median, 3) if pos.median is not None else None,
            "cohort_n": pos.n,
            "percentile": round(pos.percentile, 1) if pos.percentile is not None else None,
            "rank": pos.rank,
            "mode": pos.mode,
            "sentence": pos.describe(DISPLAY.name(m)),
            "icc": round(rel.icc, 2) if rel and rel.icc is not None else None,
            "reliable": bool(rel.trustworthy) if rel else False,
            "flag": flag,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"error": "No metric had cohort coverage.", "stratum": choice.to_dict()}

    flagged = (df[df["flag"].notna()]
                 .sort_values("percentile", ascending=False, na_position="last")
                 .reset_index(drop=True))

    # Driver scan is hypothesis generation only and is gated hard: it used to
    # run at fdr_alpha=0.20 with min_n as low as 4 and then print the top 10
    # regardless of significance, under a heading inviting coaches to treat
    # them as levers. Now it needs a real cohort and only FDR survivors count.
    driver_hits: dict[str, pd.DataFrame] = {}
    driver_note = None
    if run_drivers and not flagged.empty:
        candidates = flagged[flagged["reliable"]].head(5)
        if candidates.empty:
            driver_note = ("No flagged mechanic has an ICC high enough "
                           "(>= 0.5) to chase drivers for.")
        for _, r in candidates.iterrows():
            m = r["metric"]
            try:
                res = correlate_kinematic_to_assessments(
                    m, role="pitcher",
                    age_group=None if choice.fell_back else ctx.age_group,
                    min_trials=config.min_trials_per_athlete,
                    aggregation="mean",
                    min_n=config.min_n_cross_sectional,
                    fdr_alpha=config.fdr_alpha,
                    exclude_mobility=False,
                )
            except Exception:
                continue
            corr = res.get("correlations", pd.DataFrame())
            if corr.empty:
                continue
            sig = corr[corr["fdr_significant"]] if "fdr_significant" in corr else corr.iloc[0:0]
            if sig.empty:
                continue
            sig = sig.copy()
            sig["display_name"] = sig["assessment_metric"].map(DISPLAY.name)
            driver_hits[m] = sig.head(8).reset_index(drop=True)
        if not driver_hits and driver_note is None:
            driver_note = (
                f"No assessment cleared FDR correction as a correlate of any "
                f"flagged mechanic at n>={config.min_n_cross_sectional}. That is "
                f"the expected result at this cohort size and is reported rather "
                f"than replaced with the top of an unfiltered ranking."
            )

    return {
        "stratum": choice.to_dict(),
        "cohort_label": cohort_label,
        "percentile_threshold": config.outlier_percentile,
        "n_focus_trials": n_focus,
        "all_metrics": df,
        "flagged": flagged,
        "driver_hits_for_flags": driver_hits,
        "driver_note": driver_note,
    }


# ──────────────────────────────────────────────────────────────────────────
# Section 5: Stratum velocity drivers
# ──────────────────────────────────────────────────────────────────────────

def section_velo_drivers(ctx: AthleteContext, *,
                         config: ResearchConfig = CONFIG,
                         top_k: int = 10) -> dict:
    strat_trials, choice = loaders.resolve_stratum(
        ctx.age_group, min_athletes=config.min_athletes_for_stratum)
    if strat_trials.empty:
        return {"error": "No cohort trial data."}
    try:
        within = correlate_velocity_within_athlete(
            strat_trials,
            min_trials_per_athlete=config.min_trials_per_athlete,
            min_n=config.min_n_within_athlete,
            fdr_alpha=config.fdr_alpha,
            processed_only=True, exclude_symptomatic=True,
        )
    except Exception as e:
        return {"error": f"Within-athlete analysis failed: {e}"}
    if within.empty:
        return {"error": "No within-athlete velocity correlates met the "
                         f"n>={config.min_n_within_athlete} floor."}

    focus_mean = None
    if ctx.focus_session_date is not None and not ctx.trial_df.empty:
        f = ctx.trials_on(ctx.focus_session_date)
        if not f.empty:
            focus_mean = f.mean(numeric_only=True)

    top = within.head(top_k).copy()
    top["display_name"] = top["metric"].map(DISPLAY.name)
    top["stratum_r_vs_velo"] = top["r"].round(3)
    top["athlete_focus_value"] = [
        (round(float(focus_mean.get(m)), 3)
         if focus_mean is not None and pd.notna(focus_mean.get(m)) else None)
        for m in top["metric"]
    ]
    keep = ["metric", "display_name", "stratum_r_vs_velo"]
    for c in ("n_trials", "n_athletes", "q_value", "fdr_significant"):
        if c in top.columns:
            keep.append(c)
    keep.append("athlete_focus_value")

    return {
        "stratum": choice.to_dict(),
        "n_fdr_significant": int(within["fdr_significant"].sum())
                             if "fdr_significant" in within else 0,
        "n_tested": int(len(within)),
        "top_correlates": top[keep].reset_index(drop=True),
    }


# ──────────────────────────────────────────────────────────────────────────
# Headlines — the coach-facing distillation
# ──────────────────────────────────────────────────────────────────────────

def build_headlines(ctx: AthleteContext, fingerprint: dict, variability: dict,
                    session_change: dict, outliers: dict, velo_drivers: dict,
                    *, config: ResearchConfig = CONFIG,
                    max_per_kind: int = 4) -> tuple[list[Finding], dict]:
    """Turn the section output into a short list of sentences.

    Admission rules, applied in one place so they are auditable:
      * a CHANGE is admitted only if it cleared its measurement-error band
      * a STANDOUT is admitted only if the metric is reliable (ICC >= 0.5)
        and the cohort was big enough to place him in it
      * anything whose metric is not curated in metric_display.yaml is held
        back from the coach page — we will not put a machine-generated name
        like 'Pelvis rotation speed at ball release — axis Z' in front of a
        coach and call it a finding.
    """
    findings: list[Finding] = []
    counts: dict[str, int] = {}
    seen_metric: dict[str, int] = {}
    held_back: list[str] = []

    def admit(f: Finding) -> None:
        if not DISPLAY.is_coach_ready(f.metric):
            held_back.append(f.metric)
            return
        if counts.get(f.kind, 0) >= max_per_kind:
            return
        # The same mechanic can legitimately be a change, a limiter and an
        # instability all at once. Saying it three times, each with the same
        # coaching cue attached, reads as padding and buries the other
        # findings — so cap it at two mentions and only cue it once.
        n_seen = seen_metric.get(f.metric, 0)
        if n_seen >= 2:
            return
        if n_seen >= 1:
            f = Finding(**{**f.__dict__, "cue": None})
        seen_metric[f.metric] = n_seen + 1
        counts[f.kind] = counts.get(f.kind, 0) + 1
        findings.append(f)

    # ── Velocity, always first if we have it ──────────────────────────────
    # The comparison a coach has in mind after a session is "against last
    # time", not "against two years ago". The longer view is still reported,
    # but as context underneath rather than as the headline.
    sc: dict = session_change if isinstance(session_change, dict) else {}
    sm = sc.get("matrix")
    dv = dv_verdict = None
    if sm is not None and sm.has_comparison and not sm.frame.empty:
        vrow = sm.frame[sm.frame["metric"] == "velocity_mph"]
        if not vrow.empty and vrow.iloc[0].get("change") is not None:
            r0 = vrow.iloc[0]
            dv = float(r0["change"])
            dv_verdict = ctx.reliability.classify(
                "velocity_mph", dv,
                n_before=sm.session_trials.get(sm.previous, 1),
                n_after=sm.session_trials.get(sm.latest, 1))
    fell_back = False
    if dv is None:
        dv = sc.get("delta_velocity_first")
        dv_verdict = sc.get("delta_velocity_first_verdict")
        fell_back = dv is not None

    if dv is not None and dv_verdict is not None:
        # The label has to follow the number. When the matrix has no velocity
        # row we are quoting the first-to-last delta, and calling that "since
        # <previous capture>" is simply a false sentence.
        if sm is not None and sm.has_comparison and not fell_back:
            span = sm.days_between
            window = f"since {sm.previous}"
            long_span = bool(span and span > config.long_span_warn_days)
        else:
            span = sc.get("span_days")
            window = f"{sc.get('round1_id')} → {sc.get('roundN_id')}"
            long_span = bool(sc.get("long_span"))
        span_txt = f" ({span} days)" if span else ""
        maturation = ""
        if long_span:
            maturation = (" That gap is long enough that growth and a full "
                          "training block are part of the number.")
        # The longer arc, as context rather than as the claim.
        arc = ""
        first_dv = sc.get("delta_velocity_first")
        if (first_dv is not None and sm is not None
                and len(sm.session_dates) > 2):
            arc = (f" Across all {len(sm.session_dates)} captures he is "
                   f"{first_dv:+.1f} mph from where he started.")
        verdict = dv_verdict.verdict
        if verdict in (VERDICT_BEYOND, VERDICT_EDGE):
            word = "up" if dv > 0 else "down"
            head = f"Velocity is {word} {abs(dv):.1f} mph {window}{span_txt}."
        elif verdict == VERDICT_TYPICAL:
            head = (f"Velocity moved {dv:+.1f} mph {window}{span_txt}, smaller "
                    f"than his usual session-to-session swing.")
        else:
            head = f"Velocity moved {dv:+.1f} mph {window}{span_txt}."
        findings.append(Finding(
            kind="change", metric="velocity_mph", headline=head,
            detail=dv_verdict.explain() + maturation + arc,
            tier=TIER_STRONG if verdict == VERDICT_BEYOND else TIER_SUGGESTIVE,
            verdict=verdict, good=1 if (dv or 0) > 0 else -1,
            delta=dv, mdc=dv_verdict.mdc, group="output",
            provisional=dv_verdict.provisional,
            priority=100.0,
        ))
        counts["change"] = counts.get("change", 0) + 1

    # ── Mechanical changes, latest capture vs the one before ──────────────
    if sm is not None and not sm.frame.empty:
        big = sm.frame[(sm.frame["verdict"] == VERDICT_BEYOND)
                       & (sm.frame["metric"] != "velocity_mph")]
        for _, r in big.iterrows():
            m = r["metric"]
            good = r.get("toward_better")
            verb = r.get("change_verb", "changed")
            delta = float(r.get("change") or 0.0)
            head = (f"{DISPLAY.name(m)} {verb} by "
                    f"{DISPLAY.format_value(m, abs(delta))} since "
                    f"{sm.previous}.")
            # Put the earlier captures in the sentence — that is the whole
            # point of keeping them, and it is what tells a coach whether this
            # is new or the continuation of something.
            hist = [v for v in r.get("series", []) if v is not None and pd.notna(v)]
            hist_txt = ""
            if len(hist) > 2:
                prior = ", ".join(f"{v:,.4g}" for v in hist[:-1])
                hist_txt = (f" Across his captures: {prior} → "
                            f"{hist[-1]:,.4g}.")
            detail = (
                f"{DISPLAY.format_value(m, r.get(f's_{sm.previous}'))} → "
                f"{DISPLAY.format_value(m, r.get(f's_{sm.latest}'))}. This "
                f"measure usually varies about "
                f"{DISPLAY.format_value(m, r.get('usual_spread'))} for him, so "
                f"the move is {r.get('spread_multiple')}x its usual swing."
                f"{hist_txt} {DISPLAY.definition(m)}"
            ).strip()
            admit(Finding(
                kind="change", metric=m, headline=head, detail=detail,
                tier=TIER_STRONG, verdict=VERDICT_BEYOND, good=good,
                value=r.get(f"s_{sm.latest}"), delta=delta,
                mdc=r.get("usual_spread"),
                group=DISPLAY.group(m), cue=DISPLAY.cue(m),
                provisional=bool(r.get("provisional")),
                priority=60.0 + min(float(r.get("spread_multiple") or 0), 10.0)
                         + (5.0 if good == -1 else 0.0),
            ))

    # ── Standouts and limiters ────────────────────────────────────────────
    o = outliers if isinstance(outliers, dict) else {}
    flagged = o.get("flagged")
    if isinstance(flagged, pd.DataFrame) and not flagged.empty:
        for _, r in flagged.iterrows():
            if not r.get("reliable"):
                continue
            m = r["metric"]
            direction = r.get("direction")
            is_high = r.get("flag") == "HIGH"
            if direction == "higher_better":
                good = 1 if is_high else -1
            elif direction == "lower_better":
                good = -1 if is_high else 1
            else:
                good = None
            # A metric with no agreed good direction is not a strength or a
            # weakness — it is a trait. Saying a pitcher is "weak" at
            # foot-strike-to-release time, or "strong" at elbow varus torque,
            # is the kind of thing that erodes a coach's trust in the whole
            # report, so those get their own neutral heading instead.
            if good == 1:
                kind = "standout"
            elif good == -1:
                kind = "limiter"
            else:
                kind = "profile"
            detail = (DISPLAY.definition(m) or "").strip()
            if good is None:
                detail = (detail + " There is no agreed better direction for "
                          "this one — it is a description of how he throws, "
                          "not a score.").strip()
            admit(Finding(
                kind=kind, metric=m,
                headline=r["sentence"],
                detail=detail,
                tier=TIER_STRONG if r.get("cohort_n", 0) >= config.outlier_min_cohort_n
                     else TIER_SUGGESTIVE,
                good=good, value=r.get("athlete_value"),
                percentile=r.get("percentile"), rank=r.get("rank"),
                cohort_n=int(r.get("cohort_n") or 0),
                cohort_label=o.get("cohort_label", ""),
                group=DISPLAY.group(m), cue=DISPLAY.cue(m),
                priority=40.0 + (10.0 if good == -1 else 0.0),
            ))

    # ── Within-session credible correlates ────────────────────────────────
    v = variability if isinstance(variability, dict) else {}
    credible = v.get("credible_correlates")
    if isinstance(credible, pd.DataFrame) and not credible.empty:
        for _, r in credible.head(2).iterrows():
            m = r["metric"]
            sign = "more" if (r.get("r_vs_velo") or 0) > 0 else "less"
            admit(Finding(
                kind="variability", metric=m,
                headline=(f"On his harder pitches that day he showed {sign} "
                          f"{DISPLAY.name(m).lower()}."),
                detail=(f"Within-session Spearman r = {r['r_vs_velo']} "
                        f"{r.get('ci', '')} across {r['n_trials']} pitches. "
                        f"This is one session, so treat it as a lead to check "
                        f"next time, not a conclusion."),
                tier=TIER_SUGGESTIVE, good=None,
                group=DISPLAY.group(m), cue=DISPLAY.cue(m), priority=30.0,
            ))

    # ── Destabilised mechanics ────────────────────────────────────────────
    dstb = v.get("destabilizing_metrics")
    if isinstance(dstb, pd.DataFrame) and not dstb.empty:
        for _, r in dstb.head(2).iterrows():
            m = r["metric"]
            admit(Finding(
                kind="variability", metric=m,
                headline=(f"{DISPLAY.name(m)} was "
                          f"{r['cv_vs_hist_ratio']}× more scattered than usual "
                          f"this session."),
                detail=("Pitch-to-pitch spread this session against his own "
                        "historical spread on the same metric. A jump like "
                        "this usually means something is being changed, "
                        "fatigued, or worked around."),
                tier=TIER_SUGGESTIVE, good=-1,
                group=DISPLAY.group(m), cue=DISPLAY.cue(m), priority=35.0,
            ))

    findings.sort(key=lambda f: -f.priority)

    n_provisional = sum(1 for f in findings if f.provisional)
    summary = {
        "n_findings": len(findings),
        "n_provisional": n_provisional,
        "n_changes": sum(1 for f in findings if f.kind == "change"),
        "n_standouts": sum(1 for f in findings if f.kind == "standout"),
        "n_limiters": sum(1 for f in findings if f.kind == "limiter"),
        "n_profile_notes": sum(1 for f in findings if f.kind == "profile"),
        "held_back_uncurated": sorted(set(held_back)),
        "held_back_counts": dict(Counter(held_back)),
        "velocity_delta": dv,
        "velocity_verdict": dv_verdict.verdict if dv_verdict else None,
        "aligned_drivers": sc.get("n_aligned"),
        "joined_drivers": sc.get("n_joined"),
    }
    return findings, summary


# ──────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────

def run_athlete_deep_dive(
    query_str: str,
    *,
    focus_session_date: Any = None,
    config: ResearchConfig = CONFIG,
    outlier_percentile: float | None = None,
    min_n_cohort: int | None = None,
    top_k_velo_drivers: int = 10,
    run_drivers: bool = True,
) -> DeepDiveReport:
    """Full per-athlete research package."""
    if outlier_percentile is not None:
        config = config.replace(outlier_percentile=outlier_percentile)
    if min_n_cohort is not None:
        config = config.replace(outlier_min_cohort_n=min_n_cohort)

    uuid, name, email = resolve_athlete(query_str)
    ctx = load_athlete_context(uuid, focus_session_date=focus_session_date,
                               config=config)
    ctx.email = email or ctx.email

    warnings: list[str] = []
    if ctx.trial_df.empty:
        warnings.append("No pitching trials on record for this athlete.")
    if ctx.profile_df.empty:
        warnings.append("No athlete_profiles rows — the assessment sections "
                        "will be empty.")
    if not ctx.age_group:
        warnings.append("No age group on file, so every comparison falls back "
                        "to the whole tested population.")
    if len(ctx.reliability) == 0:
        warnings.append(
            "No measurement-reliability estimates could be computed, so no "
            "change can be checked against its noise floor. Every change "
            "below is reported as 'can't tell yet'.")
    if ctx.rounds:
        orphaned = (ctx.rounds_meta or {}).get("orphaned_assessments", {}) or {}
        n_orph = sum(len(vv) for vv in orphaned.values())
        if n_orph:
            names = ", ".join(f"{k} ({len(vv)})" for k, vv in orphaned.items())
            warnings.append(
                f"{n_orph} assessment date(s) fell outside every 3D capture's "
                f"±{(ctx.rounds_meta or {}).get('window_days', config.round_window_days)}-day "
                f"window and were excluded: {names}")
    else:
        warnings.append("No assessment rounds could be built — this athlete "
                        "has no 3D capture to anchor them to.")

    fingerprint = section_fingerprint(ctx)
    variability = section_trial_variability(ctx, config=config)
    session_change = section_session_change(ctx, config=config)
    big_picture = section_big_picture(ctx, config=config)
    outliers = section_outliers(ctx, config=config, run_drivers=run_drivers)
    coverage = section_assessment_coverage(ctx, config=config)
    velo_drivers = section_velo_drivers(ctx, config=config,
                                        top_k=top_k_velo_drivers)

    # Surface the provenance exclusions as warnings — a silently shorter
    # table is how the carry-forward bug survived this long.
    meta = session_change.get("assessment_delta_meta") if isinstance(session_change, dict) else None
    if meta:
        n_stale = len(meta.get("excluded_stale", []))
        n_same = len(meta.get("excluded_same_source", []))
        if n_stale:
            warnings.append(
                f"{n_stale} assessment metric(s) were excluded from the change "
                f"table because the value was carried forward from a session "
                f"more than {config.max_source_staleness_days} days outside the "
                f"round window — comparing those would invent change.")
        if n_same:
            warnings.append(
                f"{n_same} assessment metric(s) trace to the SAME source session "
                f"at both ends, so there is genuinely nothing to compare.")

    headlines, summary = build_headlines(
        ctx, fingerprint, variability, session_change, outliers, velo_drivers,
        config=config)

    # Scale/unit discontinuities come first — they invalidate comparisons
    # rather than merely qualifying them.
    if ctx.unit_audit is not None and ctx.unit_audit.has_findings:
        for line in ctx.unit_audit.summary_lines():
            warnings.append(line)

    mm = session_change.get("mechanic_delta_meta") if isinstance(session_change, dict) else None
    if mm:
        n_model = len(mm.get("excluded_model", {}))
        n_scale = len(mm.get("excluded_scale_break", {}))
        if n_model:
            warnings.append(
                f"{n_model} metric(s) were left out of the change tables "
                f"because they are fixed by the marker set or the capture rig "
                f"(segment lengths, frame rate, duplicate unit columns) rather "
                f"than measured from the athlete.")
        if n_scale:
            warnings.append(
                f"{n_scale} metric(s) crossed a unit or scale change between "
                f"these captures and were not differenced.")

    gaps = ctx.reliability.reliability_gaps()
    if gaps.get("needs_retest"):
        warnings.append(
            f"{len(gaps['needs_retest'])} measure(s) have no repeat captures, "
            f"so the 'usual variation' they are compared against is estimated "
            f"from pitch-to-pitch spread instead of day-to-day. That "
            f"understates day-to-day movement, so those changes look larger "
            f"than they are. Capturing a handful of athletes twice within two "
            f"weeks fixes it — run `research reliability` for the list.")

    held = summary.get("held_back_uncurated") or []
    if held:
        counts = summary.get("held_back_counts") or {}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:5] or \
            [(m, 1) for m in held[:5]]
        names = ", ".join(m for m, _ in top)
        warnings.append(
            f"{len(held)} finding(s) were kept off the coach page because "
            f"their metric has an auto-composed name rather than a curated "
            f"one. They are real measurements with a machine-written label, "
            f"so they sit in the appendix under their warehouse key. Most "
            f"frequent: {names}. Curate those in metric_display.yaml to "
            f"promote them.")

    return DeepDiveReport(
        athlete=ctx, fingerprint=fingerprint, variability=variability,
        assessment_coverage=coverage,
        session_change=session_change, big_picture=big_picture,
        outliers=outliers, velo_drivers=velo_drivers,
        headlines=headlines, summary=summary, warnings=warnings,
        config=config,
    )


__all__ = [
    "run_athlete_deep_dive", "load_athlete_context", "resolve_athlete",
    "DeepDiveReport", "AthleteContext", "SessionInfo", "ProfileSnapshot",
    "Finding", "build_headlines",
]
