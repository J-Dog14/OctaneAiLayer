"""
Single source of truth for every research threshold.

Why this exists
---------------
Before this module, `min_n`, `window_days`, `fdr_alpha`, `p_threshold` and the
CV ratio lived as function defaults scattered across six modules. Two
consequences:

  1. A finding could not be reproduced, because nobody recorded which
     thresholds produced it.
  2. Tightening a floor meant hunting every call site, and the floors
     disagreed with each other (min_n=3 in one place, 30 in another).

Now: one frozen dataclass, one hash. Every report stamps
`ResearchConfig.fingerprint()` into its header and (when persisted) into
`ai_layer.research_findings.config_hash`. If two reports disagree, the hash
tells you whether the thresholds moved or the data did.

Overriding
----------
    from src.research.config import CONFIG, ResearchConfig

    # global default
    CONFIG.min_trials_within_session

    # a one-off stricter run
    strict = CONFIG.replace(min_trials_within_session=12)

Or point RESEARCH_CONFIG_PATH at a YAML file with any subset of the fields:

    min_trials_within_session: 10
    outlier_min_cohort_n: 25
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────
# Confidence tiers — the vocabulary every renderer uses
# ──────────────────────────────────────────────────────────────────────────

TIER_STRONG = "strong"       # green  — beyond noise, adequate n, survived FDR
TIER_SUGGESTIVE = "suggestive"  # amber — real-looking but unconfirmed
TIER_INSUFFICIENT = "insufficient"  # grey — not enough data to say anything

TIER_ORDER = {TIER_STRONG: 0, TIER_SUGGESTIVE: 1, TIER_INSUFFICIENT: 2}

TIER_LABEL = {
    TIER_STRONG: "Act on this",
    TIER_SUGGESTIVE: "Worth watching",
    TIER_INSUFFICIENT: "Not enough data",
}

TIER_COLOR = {
    TIER_STRONG: "#1a7f37",
    TIER_SUGGESTIVE: "#bf8700",
    TIER_INSUFFICIENT: "#8c959f",
}


@dataclass(frozen=True)
class ResearchConfig:
    """Every knob, in one place. Frozen so nothing mutates it mid-run."""

    # ── Sample-size floors ────────────────────────────────────────────────
    # Within one session, correlating a mechanic against velocity across
    # pitches. At n=4 a |rho| of 0.8 is close to the modal outcome under the
    # null, and we scan ~150 metrics and report the max — so the old floor of
    # 4 guaranteed a table of noise. 8 is the minimum defensible; 10+ is
    # where the rank correlation starts to mean something.
    min_trials_within_session: int = 8

    # Trials an athlete needs before they contribute to a pooled/FE analysis.
    min_trials_per_athlete: int = 3

    # Residual pairs needed for a within-athlete fixed-effects correlation.
    min_n_within_athlete: int = 30

    # Athletes needed for a cross-sectional (athlete-level) correlation.
    min_n_cross_sectional: int = 12

    # Athletes needed before we quote a percentile at all. A percentile off
    # 6 observations has ~17-point resolution, so "p90" is not a real number.
    # Below this we report a rank ("3rd hardest of 9") instead.
    outlier_min_cohort_n: int = 20
    rank_min_cohort_n: int = 5

    # Athletes needed in a stratum before we use it instead of falling back
    # to the global cohort.
    min_athletes_for_stratum: int = 8

    # ── Multiple comparisons ──────────────────────────────────────────────
    fdr_alpha: float = 0.10
    # Exploratory surfaces (hypothesis generation only, never coach-facing).
    fdr_alpha_exploratory: float = 0.20

    # ── Bootstrap ─────────────────────────────────────────────────────────
    bootstrap_iterations: int = 2000
    bootstrap_ci: float = 0.90
    # Below this n, don't even bootstrap — the CI spans [-1, 1] and says
    # nothing. Report "insufficient" instead.
    bootstrap_min_n: int = 6

    # ── Reliability / minimal detectable change ───────────────────────────
    # Sessions needed before a metric gets an SEM estimate.
    reliability_min_sessions: int = 8
    # Trials within a session needed for that session to contribute to SEM.
    reliability_min_trials_per_session: int = 4
    # MDC confidence level. 1.96*sqrt(2)*SEM is the standard 95% MDC.
    mdc_z: float = 1.96
    # A delta this many multiples of MDC is "clearly real" vs "just past it".
    mdc_strong_multiple: float = 1.5

    # A between-session verdict really wants a test-retest estimate, which
    # needs repeat captures close together. Until those exist, fall back to a
    # deliberately conservative band built from within-session scatter and
    # label every verdict from it as provisional. Set False to require a real
    # test-retest estimate and report "can't tell yet" everywhere else —
    # stricter, and the right setting once repeat captures are routine.
    allow_provisional_band: bool = True

    # ── Assessment rounds ─────────────────────────────────────────────────
    # A round pairs a 3D capture with the assessments around it. The window has
    # to match how the facility actually tests: 3D runs a couple of times a
    # year, screens quarterly, Proteus weekly. At +/-14 days a real athlete had
    # 49 of his assessment dates fall outside every window — most of his
    # testing was invisible to the analysis. 45 days keeps a round inside one
    # training block while catching the screen that bracketed the capture.
    # Assessments that still fall outside every window are listed by
    # `assessment_coverage` rather than silently dropped.
    #
    # 60, not 45: a real athlete screened on 2025-09-03 and threw his 3D on
    # 2025-10-24 — 51 days, which fell outside a 45-day window and lost the
    # drop jump, plyo push-up and single-leg vertical from that round. This
    # window ONLY governs which assessments get paired with a capture for the
    # mechanics-linking analysis. His assessment history is no longer gated on
    # it at all: that table is built from his own test dates.
    round_window_days: int = 60
    # A profile metric whose source session is older than this, relative to
    # the round anchor, is a carry-forward and must not be differenced as if
    # it were measured in that round.
    max_source_staleness_days: int = 45
    # Flag any round-to-round comparison spanning longer than this. Over a
    # year of HS development, a chunk of any delta is maturation.
    long_span_warn_days: int = 180

    # ── Variability ───────────────────────────────────────────────────────
    # CV ratio (this session vs historical) above which a metric is flagged
    # as destabilizing.
    destabilizing_cv_ratio: float = 1.5
    # Historical trials needed before a historical CV baseline is trustworthy.
    min_hist_trials_for_cv: int = 12

    # ── Display ───────────────────────────────────────────────────────────
    outlier_percentile: float = 90.0
    top_k_default: int = 10
    # Never show a coach a table longer than this without a details wrapper.
    max_coach_table_rows: int = 8

    # ── Provenance ────────────────────────────────────────────────────────
    # Bumped by hand when the meaning of a computation changes (not just a
    # threshold). Lets you invalidate cached findings.
    analysis_version: str = "2.0"

    # ──────────────────────────────────────────────────────────────────────

    @property
    def mdc_multiplier(self) -> float:
        """MDC = mdc_multiplier * SEM. The sqrt(2) is because a change score
        is a difference of two measurements, each carrying its own error."""
        return self.mdc_z * (2 ** 0.5)

    def replace(self, **kwargs: Any) -> "ResearchConfig":
        """A copy with some fields overridden. Validates as it goes."""
        unknown = set(kwargs) - {f for f in asdict(self)}
        if unknown:
            raise ValueError(f"Unknown config field(s): {sorted(unknown)}")
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of the full config. Stamp this on every output."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def describe(self) -> str:
        """One-line human summary for report headers."""
        return (
            f"config {self.fingerprint()} · v{self.analysis_version} · "
            f"within-session n>={self.min_trials_within_session} · "
            f"FE n>={self.min_n_within_athlete} · "
            f"cohort n>={self.outlier_min_cohort_n} · "
            f"FDR alpha={self.fdr_alpha}"
        )


def _load_overrides() -> dict[str, Any]:
    path = os.getenv("RESEARCH_CONFIG_PATH")
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"RESEARCH_CONFIG_PATH={path} does not exist.")
    import yaml  # local import — only needed when an override file is used
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


CONFIG: ResearchConfig = ResearchConfig(**_load_overrides())


__all__ = [
    "CONFIG",
    "ResearchConfig",
    "TIER_STRONG",
    "TIER_SUGGESTIVE",
    "TIER_INSUFFICIENT",
    "TIER_ORDER",
    "TIER_LABEL",
    "TIER_COLOR",
]
