"""
What kind of number is this, and does it belong in a change table?

Two jobs.

1. EXCLUDE things that are not measurements of the athlete's movement.
   The change table for Ryan Chasse led with:

       MODEL RTA Seg Length X   0.540 -> 0.559   (+0.019)   "real"
       MODEL LTH Seg Length X   0.459 -> 0.440   (-0.019)   "real"
       Body weight n            1130  -> 1130    (+0.388)   "real"

   Segment lengths are model constants derived from marker placement. They
   change when the tech puts a marker somewhere slightly different, not when
   the pitcher changes. Worse, they are near-constant WITHIN a session, so
   their measured spread is ~0, and dividing a real between-session difference
   by ~0 produced ratios of 7.79e+14 — which sorted them straight to the top
   of the table. Anything constant by construction has to be out of the change
   analysis entirely, not merely down-ranked.

   Bodyweight is kept (it matters) but only once: kg, not the newtons twin.

2. SPLIT kinetics from kinematics.
   Force-plate metrics and joint-angle metrics answer different questions and
   there are far more force metrics in the warehouse, so a single table ranked
   by magnitude buries the kinematics behind the kinetics every time. They get
   their own tables so both are visible.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import pandas as pd

# ── Families ──────────────────────────────────────────────────────────────
KINETICS = "kinetics"        # forces, torques, impulses — what he pushed with
KINEMATICS = "kinematics"    # angles, velocities, positions — how he moved
TIMING = "timing"            # event times and durations
OUTPUT = "output"            # velocity, the thing being explained
ANTHRO = "anthropometric"    # bodyweight, height — context, not mechanics
MODEL = "model"              # segment lengths, frame rate — rig constants
OTHER = "other"

FAMILY_LABEL = {
    KINETICS: "Kinetics — forces and torques",
    KINEMATICS: "Kinematics — how he moved",
    TIMING: "Timing",
    OUTPUT: "Output",
    ANTHRO: "Body",
    MODEL: "Model / rig",
    OTHER: "Other",
}
FAMILY_ORDER = [OUTPUT, KINETICS, KINEMATICS, TIMING, ANTHRO, MODEL, OTHER]


# ── Exclusions ────────────────────────────────────────────────────────────

# Constant by construction: a property of the marker set, the model or the
# capture rig, not of the athlete's movement on the day.
_MODEL_PATTERNS = (
    r"seg_length",
    r"^kin_model\.",
    r"frame_rate",
    r"framerate",
    r"plate_id",
    r"processor_version",
)

# Same measurement carried twice in different units. Keep one.
_REDUNDANT_UNIT_TWINS = {
    # keep -> drop
    "fm_body_weight_kg": ["fm_body_weight_n"],
}

# Event indices, flags and direction markers — not physiological.
_NON_PHYSIOLOGICAL = (
    r"_frame$", r"_frame_", r"^fm_.*_frame$",
    r"_flag$", r"_sign$",
    r"_id$",
    # Absolute clock times inside a capture. `fm_fc_time_s` is not "when foot
    # strike happened" in any sense a coach cares about — it is where the
    # event fell on a clock that started whenever recording started. One real
    # report differenced these and reported +0.409 s on "front leg peak
    # vertical force", which measured when somebody hit record. The intervals
    # (`_duration_s`, `_lag_ms`, `foot_loading_time_ms`) are the real timing
    # metrics and are untouched by this.
    r"^fm_.*_time_s$",
    # Which lab axis the processor called vertical/braking on this trial.
    r"^fm_axis_",
)

_MODEL_RE = re.compile("|".join(_MODEL_PATTERNS), re.I)
_NONPHYS_RE = re.compile("|".join(_NON_PHYSIOLOGICAL), re.I)

_DROPPED_TWINS = {d for lst in _REDUNDANT_UNIT_TWINS.values() for d in lst}


def is_model_constant(metric: str) -> bool:
    """True for anything fixed by the rig or the model rather than measured."""
    return bool(_MODEL_RE.search(metric))


def is_non_physiological(metric: str) -> bool:
    return bool(_NONPHYS_RE.search(metric))


def is_redundant_unit_twin(metric: str) -> bool:
    return metric in _DROPPED_TWINS


def exclude_from_change(metric: str) -> str | None:
    """Reason this metric must not appear in a change table, or None.

    Returned as a reason string rather than a bool so the report can say what
    it left out and why, instead of silently showing a shorter table.
    """
    if is_model_constant(metric):
        return ("fixed by the marker set / capture rig — changes with marker "
                "placement, not with the athlete")
    if is_non_physiological(metric):
        return "an index, flag or identifier, not a measurement"
    if is_redundant_unit_twin(metric):
        keep = next(k for k, v in _REDUNDANT_UNIT_TWINS.items() if metric in v)
        return f"same measurement as {keep}, in different units"
    return None


def filter_for_change(metrics: Iterable[str]) -> tuple[list[str], dict[str, str]]:
    """(kept, {dropped_metric: reason})."""
    kept, dropped = [], {}
    for m in metrics:
        why = exclude_from_change(m)
        if why:
            dropped[m] = why
        else:
            kept.append(m)
    return kept, dropped


# ── Family classification ─────────────────────────────────────────────────

_KINETIC_TOKENS = (
    "grf", "force", "torque", "impulse", "rfd", "moment", "loading",
    "braking", "vertical_bw", "peak_v", "_bw", "_n_per_s", "newton",
)
_KINEMATIC_TOKENS = (
    "angle", "ang_vel", "ang_acc", "rotation", "abduction", "adduction",
    "flexion", "extension", "separation", "sep_", "cog", "position",
    "stride", "knee", "pelvis", "trunk", "thorax", "shoulder", "elbow",
    "hand", "humerus", "obliquity", "tilt", "lean", "linear_vel",
)
_TIMING_TOKENS = (
    "time", "_ms", "duration", "lag", "handoff", "timing", "_s$",
)
_ANTHRO_TOKENS = ("body_weight", "bodyweight", "height", "mass")


def classify_family(metric: str) -> str:
    """Which table does this belong in?

    Ordering matters: a metric can match several token sets ('front-leg peak
    vertical force' contains both a force token and a leg token), so the more
    specific family wins. Kinetics is checked before kinematics because force
    metric names very often carry a segment name too.
    """
    m = metric.lower()

    if metric in ("velocity_mph", "pitch_ball_release_speed"):
        return OUTPUT
    if is_model_constant(metric):
        return MODEL
    if any(t in m for t in _ANTHRO_TOKENS):
        return ANTHRO

    # A force-plate prefix is decisive — that table is all kinetics.
    if m.startswith("fm_"):
        # ...except the timing columns that live in it.
        if any(t in m for t in ("time", "_ms", "duration", "lag", "handoff")):
            return TIMING
        return KINETICS

    if any(t in m for t in _KINETIC_TOKENS):
        return KINETICS
    if any(re.search(t, m) for t in _TIMING_TOKENS):
        return TIMING
    if any(t in m for t in _KINEMATIC_TOKENS):
        return KINEMATICS
    return OTHER


def split_by_family(metrics: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for m in metrics:
        out.setdefault(classify_family(m), []).append(m)
    return out


def add_family_column(df: pd.DataFrame, *, metric_col: str = "metric"
                      ) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["family"] = out[metric_col].map(classify_family)
    return out


__all__ = [
    "KINETICS", "KINEMATICS", "TIMING", "OUTPUT", "ANTHRO", "MODEL", "OTHER",
    "FAMILY_LABEL", "FAMILY_ORDER",
    "classify_family", "split_by_family", "add_family_column",
    "exclude_from_change", "filter_for_change",
    "is_model_constant", "is_non_physiological", "is_redundant_unit_twin",
]
