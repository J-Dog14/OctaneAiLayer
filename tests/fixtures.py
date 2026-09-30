"""
Synthetic warehouse for testing the research stack without a database.

Two things this buys us:

  1. A NULL CONTROL. Generate trials where velocity is unrelated to every
     mechanic, run the whole pipeline, and assert it reports nothing. Any
     analysis that manufactures findings out of noise fails here loudly. This
     is the test that would have caught the old within-session table.

  2. A RECOVERY CONTROL. Plant a known relationship of known strength and a
     known measurement-noise level, and assert the pipeline finds the planted
     one, estimates the noise floor close to the truth, and does not also
     "find" the decoys.

The generator mirrors the real shape of the warehouse: athletes have several
sessions, sessions have several pitches, each metric has a true per-athlete
level plus per-session drift plus per-trial measurement noise. That three-level
structure is what makes within-athlete and between-athlete analyses behave
differently, so a fixture without it would not exercise the code that matters.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd

# Trial-level metric names chosen to survive the real column filters in
# `metric_columns_pitching` and to have curated display entries.
CURATED_TRIAL_METRICS = [
    "fm_lead_peak_vertical_bw",
    "fm_lead_peak_braking_bw",
    "fm_lead_rfd_vertical_bw_per_s",
    "fm_lead_rfd_braking_bw_per_s",
    "fm_lead_time_to_peak_fz_ms",
    "fm_peak_v_to_peak_b_lag_ms",
    "fm_lead_impulse_v_into_ball_bws",
    "fm_drive_peak_vertical_bw",
    "fm_fc_to_br_duration_s",
]
UNCURATED_TRIAL_METRICS = [
    "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z",
    "kin_PROCESSED.Trunk_Angle@Footstrike.Y",
    "kin_PROCESSED.Lead_Knee_Angle@Release.X",
    "kin_PROCESSED.Pelvis_Angle@Footstrike.Z",
]
ALL_TRIAL_METRICS = CURATED_TRIAL_METRICS + UNCURATED_TRIAL_METRICS

# Profile-level (assessment) metrics with curated display entries.
PROFILE_METRICS = [
    "mob_shoulder_ir",
    "mob_shoulder_er",
    "mob_l_prone_hip_ir",
    "screen_dj_rsi",
    "screen_cmj_jh_in",
    "proteus_pitcher_d2_extension_power_mean",
]
# Which modality each profile metric comes from — mirrors metrics_spec so the
# staleness gating in the deep dive has something real to check.
PROFILE_MODALITY = {
    "mob_shoulder_ir": "mobility",
    "mob_shoulder_er": "mobility",
    "mob_l_prone_hip_ir": "mobility",
    "screen_dj_rsi": "athletic_screen_dj",
    "screen_cmj_jh_in": "athletic_screen_cmj",
    "proteus_pitcher_d2_extension_power_mean": "proteus_pitcher",
}


@dataclass
class PlantedEffect:
    """A known truth for the pipeline to find (or fail to find)."""
    metric: str
    within_athlete_beta: float = 0.0   # mph per unit, within an athlete
    between_athlete_beta: float = 0.0  # mph per unit, across athletes


@dataclass
class SyntheticWarehouse:
    trials: pd.DataFrame
    profiles_wide: pd.DataFrame
    snapshots: dict[str, list[dict]]        # athlete_uuid -> profile rows
    rounds: dict[str, tuple[list, dict]]    # athlete_uuid -> (rounds, meta)
    athletes: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)

    def uuid_for(self, name: str) -> str:
        row = self.athletes[self.athletes["name"] == name]
        if row.empty:
            raise KeyError(name)
        return row.iloc[0]["athlete_uuid"]


def make_warehouse(
    *,
    n_athletes: int = 14,
    sessions_per_athlete: int = 3,
    trials_per_session: int = 12,
    planted: Iterable[PlantedEffect] = (),
    measurement_sd: dict[str, float] | None = None,
    base_velocity: float = 86.0,
    velocity_noise_sd: float = 1.2,
    seed: int = 7,
    start: date = date(2025, 1, 15),
    session_gap_days: int = 120,
    retest_athletes: int = 0,
    retest_gap_days: int = 7,
    session_drift_scale: float = 0.35,
) -> SyntheticWarehouse:
    """Build a three-level synthetic warehouse.

    Each metric value = athlete level + session drift + trial noise, where the
    trial noise SD is what `reliability.compute_reliability` should recover as
    the SEM. Velocity is built from the planted effects plus its own noise, so
    a within-athlete analysis has something real to find and a between-athlete
    analysis has a different something.

    `session_drift_scale` is the day-to-day component — the thing that makes
    within-session scatter an underestimate of real test-retest error. Set it
    to 0 for a world where the only noise is pitch-to-pitch.

    `retest_athletes` gives that many athletes an extra capture
    `retest_gap_days` after their first, which is what
    `estimate_between_session` needs to produce a real day-to-day band. Mirrors
    the operational recommendation: a handful of quick repeat captures unlocks
    verdicts for everyone.
    """
    rng = np.random.default_rng(seed)
    planted = list(planted)
    planted_by_metric = {p.metric: p for p in planted}
    measurement_sd = measurement_sd or {}

    # Per-metric scale so the synthetic numbers are not all N(0,1).
    scale = {m: float(rng.uniform(4.0, 30.0)) for m in ALL_TRIAL_METRICS}
    centre = {m: float(rng.uniform(10.0, 200.0)) for m in ALL_TRIAL_METRICS}
    noise_sd = {m: float(measurement_sd.get(m, scale[m] * 0.25))
                for m in ALL_TRIAL_METRICS}

    athletes = []
    for i in range(n_athletes):
        athletes.append({
            "athlete_uuid": f"{i:08d}-0000-4000-8000-{i:012d}",
            "name": f"Athlete {i:02d}",
            "email": f"athlete{i:02d}@example.test",
            "age_group": ["HIGH SCHOOL", "COLLEGE", "PRO"][i % 3],
            "handedness": "R" if i % 4 else "L",
            "height": float(72 + rng.normal(0, 2)),
            "weight": float(190 + rng.normal(0, 15)),
        })
    ath_df = pd.DataFrame(athletes)

    # Per-athlete true levels
    athlete_level = {
        m: dict(zip(ath_df["athlete_uuid"],
                    centre[m] + rng.normal(0, scale[m], size=n_athletes)))
        for m in ALL_TRIAL_METRICS
    }
    athlete_velo_level = dict(zip(
        ath_df["athlete_uuid"], base_velocity + rng.normal(0, 4.0, size=n_athletes)))

    rows: list[dict] = []
    for _, a in ath_df.iterrows():
        uuid = a["athlete_uuid"]
        for s in range(sessions_per_athlete):
            sdate = start + timedelta(days=s * session_gap_days)
            drift = {m: float(rng.normal(0, scale[m] * session_drift_scale))
                     for m in ALL_TRIAL_METRICS}
            for t in range(trials_per_session):
                rec: dict[str, Any] = {
                    "athlete_uuid": uuid,
                    "name": a["name"],
                    "session_date": sdate,
                    "trial_index": t,
                    "age_at_collection": 19.0 + s * 0.3,
                    "age_group": a["age_group"],
                    "height": a["height"],
                    "weight": a["weight"],
                    "handedness": a["handedness"],
                    "score": float(rng.normal(50, 5)),
                }
                for m in ALL_TRIAL_METRICS:
                    rec[m] = (athlete_level[m][uuid] + drift[m]
                              + rng.normal(0, noise_sd[m]))
                # Velocity from the planted effects.
                v = athlete_velo_level[uuid] + rng.normal(0, velocity_noise_sd)
                for p in planted:
                    if p.within_athlete_beta:
                        within_dev = rec[p.metric] - (athlete_level[p.metric][uuid])
                        v += p.within_athlete_beta * within_dev
                    if p.between_athlete_beta:
                        v += p.between_athlete_beta * (
                            athlete_level[p.metric][uuid] - centre[p.metric])
                rec["velocity_mph"] = float(v)
                rows.append(rec)

    # Retest captures: a second look a few days later, close enough that no
    # real adaptation could have happened, which is what a test-retest
    # reliability estimate is built from.
    for a_i in range(min(retest_athletes, n_athletes)):
        a = ath_df.iloc[a_i]
        uuid = a["athlete_uuid"]
        sdate = start + timedelta(days=retest_gap_days)
        drift = {m: float(rng.normal(0, scale[m] * session_drift_scale))
                 for m in ALL_TRIAL_METRICS}
        for t in range(trials_per_session):
            rec = {
                "athlete_uuid": uuid, "name": a["name"], "session_date": sdate,
                "trial_index": t, "age_at_collection": 19.0,
                "age_group": a["age_group"], "height": a["height"],
                "weight": a["weight"], "handedness": a["handedness"],
                "score": float(rng.normal(50, 5)),
            }
            for m in ALL_TRIAL_METRICS:
                rec[m] = (athlete_level[m][uuid] + drift[m]
                          + rng.normal(0, noise_sd[m]))
            rec["velocity_mph"] = float(
                athlete_velo_level[uuid] + rng.normal(0, velocity_noise_sd))
            rows.append(rec)

    trials = pd.DataFrame(rows)
    # A couple of columns the real loader always carries and the filters drop.
    trials["fm_fc_frame"] = rng.integers(10, 90, size=len(trials))
    trials["fm_braking_sign"] = 1

    # ── Profiles: one snapshot per session date, per athlete ──────────────
    prof_scale = {m: float(rng.uniform(5.0, 40.0)) for m in PROFILE_METRICS}
    prof_centre = {m: float(rng.uniform(20.0, 120.0)) for m in PROFILE_METRICS}
    snapshots: dict[str, list[dict]] = {}
    wide_rows: list[dict] = []
    pid = 1
    for _, a in ath_df.iterrows():
        uuid = a["athlete_uuid"]
        snaps = []
        for s in range(sessions_per_athlete):
            sdate = start + timedelta(days=s * session_gap_days)
            raw = {m: float(prof_centre[m] + rng.normal(0, prof_scale[m]))
                   for m in PROFILE_METRICS}
            z = {m: float((raw[m] - prof_centre[m]) / prof_scale[m])
                 for m in PROFILE_METRICS}
            source_dates = {PROFILE_MODALITY[m]: sdate.isoformat()
                            for m in PROFILE_METRICS}
            source_dates["pitching"] = sdate.isoformat()
            snaps.append({
                "id": pid, "as_of_date": sdate, "age_group": a["age_group"],
                "z_scores": z, "raw_values": raw, "source_dates": source_dates,
            })
            wide_rows.append({
                "profile_id": pid, "athlete_uuid": uuid, "name": a["name"],
                "age_group": a["age_group"], "as_of_date": sdate,
                "has_pitching_data": True, "has_hitting_data": False,
                "source_dates": source_dates, "role": "pitcher", **z,
            })
            pid += 1
        snapshots[uuid] = snaps

    profiles_wide = pd.DataFrame(wide_rows)
    latest = (profiles_wide.sort_values("as_of_date")
                           .groupby("athlete_uuid", as_index=False).tail(1)
                           .reset_index(drop=True))

    # ── Rounds: one per session, anchored on the 3D capture ───────────────
    from src.research.assessment_rounds import AssessmentRound
    rounds: dict[str, tuple[list, dict]] = {}
    for _, a in ath_df.iterrows():
        uuid = a["athlete_uuid"]
        rs = []
        for s in range(sessions_per_athlete):
            sdate = start + timedelta(days=s * session_gap_days)
            rs.append(AssessmentRound(
                round_id=f"R{s + 1}", anchor_type="pitching_3d",
                anchor_date=sdate,
                window_start=sdate - timedelta(days=14),
                window_end=sdate + timedelta(days=14),
                sources_present={"mobility": sdate, "screen_dj": sdate,
                                 "proteus": sdate},
                profile_id=snapshots[uuid][s]["id"],
                profile_as_of_date=sdate,
            ))
        rounds[uuid] = (rs, {"anchor_type": "pitching_3d", "n_anchors": len(rs),
                             "n_rounds": len(rs), "orphaned_assessments": {},
                             "window_days": 14})

    return SyntheticWarehouse(
        trials=trials, profiles_wide=latest, snapshots=snapshots,
        rounds=rounds, athletes=ath_df,
        truth={
            "planted": {p.metric: p for p in planted},
            "noise_sd": noise_sd,
            "scale": scale,
            "trials_per_session": trials_per_session,
            "sessions_per_athlete": sessions_per_athlete,
            "n_athletes": n_athletes,
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Patching
# ──────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def patched_warehouse(wh: SyntheticWarehouse):
    """Swap every warehouse entry point for the synthetic data.

    Patching at the loader boundary (rather than mocking psycopg2) keeps the
    tests exercising all of the real analysis code — filters, aggregation,
    reliability, tiering, rendering — and only the I/O is fake.
    """
    import src.research.athlete_deep_dive as add
    import src.research.assessment_rounds as ar
    import src.research.kinematic_drivers as kd
    import src.research.loaders as ld
    import src.research.pitching_deep as pdeep
    import src.research.profile_matrix as pm

    originals = {
        "pd_load": pdeep.load_pitching_trials_wide,
        "pm_load": pm.load_matrix,
        "kd_load": kd.load_pitching_trials_wide,
        "kd_matrix": kd.load_matrix,
        "add_snaps": add._load_profile_snapshots,
        "add_rounds": add.build_assessment_rounds,
        "add_resolve": add.resolve_athlete,
        "add_conn": add.backend_conn,
        "add_query": add.query,
        "ar_build": ar.build_assessment_rounds,
    }

    def fake_trials(*, age_group=None, min_velocity=None, max_velocity=None,
                    include_force_metrics=True, min_as_of_date=None):
        df = wh.trials.copy()
        if age_group:
            df = df[df["age_group"].str.strip().str.upper()
                    == str(age_group).strip().upper()]
        if min_velocity is not None:
            df = df[df["velocity_mph"] >= min_velocity]
        if max_velocity is not None:
            df = df[df["velocity_mph"] <= max_velocity]
        return df.reset_index(drop=True)

    def fake_matrix(*, role=None, age_group=None, latest_only=True,
                    require_modalities=None, min_non_null_metrics=5,
                    include_raw=False, min_as_of_date=None,
                    exclude_mobility=False):
        df = wh.profiles_wide.copy()
        if age_group:
            df = df[df["age_group"] == age_group]
        if exclude_mobility:
            df = df.drop(columns=[c for c in df.columns if c.startswith("mob_")],
                         errors="ignore")
        return df.reset_index(drop=True)

    def fake_snapshots(uuid):
        from src.research.athlete_deep_dive import ProfileSnapshot
        snaps = wh.snapshots.get(uuid, [])
        if not snaps:
            return pd.DataFrame(), {}
        recs, out = [], {}
        for s in snaps:
            rec = {"profile_id": s["id"], "as_of_date": s["as_of_date"],
                   "age_group": s["age_group"], **s["z_scores"]}
            recs.append(rec)
            out[s["id"]] = ProfileSnapshot(
                profile_id=s["id"], as_of_date=s["as_of_date"],
                age_group=s["age_group"], z_scores=dict(s["z_scores"]),
                raw_values=dict(s["raw_values"]),
                source_dates=dict(s["source_dates"]),
            )
        return pd.DataFrame(recs), out

    def fake_rounds(uuid, *, window_days=14, prefer_hitting=False):
        return wh.rounds.get(uuid, ([], {"orphaned_assessments": {},
                                         "window_days": window_days}))

    def fake_resolve(q):
        m = wh.athletes[(wh.athletes["name"] == q)
                        | (wh.athletes["athlete_uuid"] == q)
                        | (wh.athletes["email"] == q)]
        if m.empty:
            raise ValueError(f"No athlete matched {q!r}")
        r = m.iloc[0]
        return r["athlete_uuid"], r["name"], r["email"]

    class _FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_conn():
        return _FakeConn()

    def fake_query(conn, sql, params=None):
        # Only the email/name lookup goes through here in the deep dive.
        if params:
            m = wh.athletes[wh.athletes["athlete_uuid"] == params[0]]
            if not m.empty:
                r = m.iloc[0]
                return [{"name": r["name"], "email": r["email"]}]
        return []

    pdeep.load_pitching_trials_wide = fake_trials
    kd.load_pitching_trials_wide = fake_trials
    pm.load_matrix = fake_matrix
    kd.load_matrix = fake_matrix
    add._load_profile_snapshots = fake_snapshots
    add.build_assessment_rounds = fake_rounds
    ar.build_assessment_rounds = fake_rounds
    add.resolve_athlete = fake_resolve
    add.backend_conn = fake_conn
    add.query = fake_query
    ld.clear_cache()
    try:
        yield wh
    finally:
        pdeep.load_pitching_trials_wide = originals["pd_load"]
        pm.load_matrix = originals["pm_load"]
        kd.load_pitching_trials_wide = originals["kd_load"]
        kd.load_matrix = originals["kd_matrix"]
        add._load_profile_snapshots = originals["add_snaps"]
        add.build_assessment_rounds = originals["add_rounds"]
        ar.build_assessment_rounds = originals["ar_build"]
        add.resolve_athlete = originals["add_resolve"]
        add.backend_conn = originals["add_conn"]
        add.query = originals["add_query"]
        ld.clear_cache()
