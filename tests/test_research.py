"""
Tests for the research stack. No database required — everything runs against
the synthetic warehouse in `tests/fixtures.py`.

    pytest tests/ -v

The two that matter most:

  test_null_warehouse_yields_no_findings
      Pure noise in, nothing out. An analysis pipeline that invents findings
      from random data is worse than no pipeline, because the findings look
      exactly like the real ones.

  test_planted_effect_is_recovered
      The complement. Having proved it says nothing when there is nothing,
      prove it still says something when there is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.config import CONFIG, ResearchConfig
from src.research.metric_display import (
    DISPLAY,
    STATUS_CURATED,
    STATUS_RAW,
)
from src.research.reliability import (
    COMPARISON_BETWEEN,
    COMPARISON_WITHIN,
    SOURCE_BETWEEN,
    SOURCE_WITHIN,
    VERDICT_NOISE,
    VERDICT_REAL,
    VERDICT_UNKNOWN,
    ReliabilityTable,
    compute_reliability,
    estimate_between_session,
    merge_reliability,
)

from src.research.stats_support import (
    eb_shrink_means,
    fisher_shrink_r,
    max_abs_r_null,
    percentile_or_rank,
    spearman_with_ci,
)
from tests.fixtures import (
    ALL_TRIAL_METRICS,
    CURATED_TRIAL_METRICS,
    PlantedEffect,
    make_warehouse,
    patched_warehouse,
)
def _rel(sem: float = 5.0, source: str = SOURCE_BETWEEN,
         icc: float = 0.8) -> ReliabilityTable:
    """A one-metric reliability table for the arithmetic tests."""
    return ReliabilityTable(pd.DataFrame([{
        "metric": "m", "sem": sem, "cv_pct": None, "icc": icc,
        "mdc95_single": CONFIG.mdc_multiplier * sem, "sd_between": sem * 2,
        "n_sessions": 40, "n_trials": 480, "n_athletes": 20,
        "source": source,
    }]))


# ══════════════════════════════════════════════════════════════════════════
# The null control
# ══════════════════════════════════════════════════════════════════════════

def test_null_warehouse_yields_no_findings():
    """Velocity unrelated to every mechanic → no 'real' mechanical findings.

    The old within-session section would happily report the largest of ~150
    coin flips as the top velocity correlate. This asserts the new one does
    not, across several seeds so it is not a single lucky draw.
    """
    for seed in (1, 2, 3):
        wh = make_warehouse(n_athletes=12, sessions_per_athlete=2,
                            trials_per_session=14, planted=(),
                            retest_athletes=10, seed=seed)
        with patched_warehouse(wh):
            from src.research.athlete_deep_dive import run_athlete_deep_dive
            rep = run_athlete_deep_dive("Athlete 00")

            credible = rep.variability.get("credible_correlates")
            n_credible = 0 if credible is None or credible.empty else len(credible)
            assert n_credible == 0, (
                f"seed {seed}: reported {n_credible} credible within-session "
                f"velocity correlates from pure noise")

            # Nothing may be promoted to a REAL mechanical change. Amber
            # ("likely") is allowed to fire on noise by design — it is an
            # 80% band, so roughly one unchanged measurement in five lands
            # there, and the page words it as "worth confirming", not as a
            # finding. The calibration of both tiers is asserted separately
            # in test_verdict_false_positive_rate_is_calibrated.
            v = rep.summary.get("velocity_verdict")
            assert v != VERDICT_REAL, (
                f"seed {seed}: velocity change called real on null data")

            mech = rep.session_change.get("mechanic_deltas")
            if isinstance(mech, pd.DataFrame) and not mech.empty:
                n_real = int((mech["verdict"] == VERDICT_REAL).sum())
                rate = n_real / len(mech)
                assert rate <= 0.20, (
                    f"seed {seed}: {n_real}/{len(mech)} mechanics called real "
                    f"changes on null data ({rate:.0%}) — the noise floor is "
                    f"not doing its job")


def test_null_data_top_correlation_is_not_promoted():
    """Even when the scan produces a big |r| by chance, it must not clear the
    scan-wide noise ceiling."""
    wh = make_warehouse(n_athletes=8, sessions_per_athlete=1,
                        trials_per_session=9, planted=(), seed=11)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        v = rep.variability
        if v.get("suppressed"):
            return  # suppression is also a correct answer
        top = v["top_velo_correlates"]
        assert not top.empty
        biggest = top["r_vs_velo"].abs().max()
        ceiling = v["noise_ceiling"].null_p95
        credible = v.get("credible_correlates")
        if biggest < ceiling:
            assert credible is None or credible.empty
        # And whatever happened, the section states the ceiling out loud.
        assert "noise" in v["noise_ceiling"].explain().lower()


def test_pitch_to_pitch_runs_at_any_n():
    """Every measurement is data. With four pitches we still report what each
    pitch did and how far apart they were — only the ranked correlation scan
    is flagged as underpowered."""
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=2,
                        trials_per_session=4, seed=5)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        v = rep.variability
        assert "error" not in v, v.get("error")
        assert v["n_trials"] == 4
        assert not v["all_metrics"].empty
        # per-pitch values are present, not summarised away
        assert len(v["per_pitch"]) == 4
        # spread is reported for every metric
        assert v["all_metrics"]["range"].notna().all()
        # the scan is labelled, not suppressed
        assert v["correlations_underpowered"] is True
        assert v["correlation_floor"] == CONFIG.min_trials_within_session


def test_two_pitches_reports_the_difference():
    """'If there are 2 pitches, I want to see the difference between them.'"""
    wh = make_warehouse(n_athletes=8, sessions_per_athlete=2,
                        trials_per_session=2, seed=12)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        v = rep.variability
        assert "error" not in v
        assert v["n_trials"] == 2
        am = v["all_metrics"]
        assert am["pitch_to_pitch_diff"].notna().any()
        # and the difference equals max - min for a two-pitch session
        row = am[am["pitch_to_pitch_diff"].notna()].iloc[0]
        assert abs(abs(row["pitch_to_pitch_diff"]) - row["range"]) < 1e-6


# ══════════════════════════════════════════════════════════════════════════
# Recovery
# ══════════════════════════════════════════════════════════════════════════

def test_planted_effect_is_recovered():
    """A planted within-athlete velocity driver shows up at the top of the
    cohort analysis, and the decoys do not."""
    target = "fm_lead_rfd_braking_bw_per_s"
    wh = make_warehouse(n_athletes=16, sessions_per_athlete=3,
                        trials_per_session=14,
                        planted=[PlantedEffect(target, within_athlete_beta=0.30)],
                        seed=21)
    with patched_warehouse(wh):
        from src.research.pitching_deep import correlate_velocity_within_athlete
        res = correlate_velocity_within_athlete(
            wh.trials, min_trials_per_athlete=3, min_n=30,
            processed_only=True, exclude_symptomatic=True)
        assert not res.empty
        assert res.iloc[0]["metric"] == target, (
            f"expected the planted metric on top, got "
            f"{res.iloc[0]['metric']} (r={res.iloc[0]['r']:.2f})")
        assert bool(res.iloc[0]["fdr_significant"])
        assert abs(res.iloc[0]["r"]) > 0.3

        others = res[res["metric"] != target]
        n_sig = int(others["fdr_significant"].sum())
        assert n_sig <= 2, f"{n_sig} decoys also cleared FDR"


def test_planted_effect_reaches_the_coach_page():
    """End to end: a real change produces a coach-ready finding with a verdict
    and a plain-language name."""
    wh = make_warehouse(n_athletes=14, sessions_per_athlete=3,
                        trials_per_session=14,
                        planted=[PlantedEffect("fm_lead_rfd_braking_bw_per_s",
                                               within_athlete_beta=0.30)],
                        seed=33)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 03")
        assert rep.headlines, "no findings at all on data with real movement"
        for f in rep.headlines:
            assert DISPLAY.is_coach_ready(f.metric), (
                f"{f.metric} reached the coach page without a curated name")
            assert f.headline and not f.headline.startswith("None")
            assert f.tier in ("strong", "suggestive", "insufficient")


# ══════════════════════════════════════════════════════════════════════════
# Reliability
# ══════════════════════════════════════════════════════════════════════════

def test_between_session_estimate_exceeds_within_session():
    """With real day-to-day drift in the data, the test-retest band must come
    out wider than trial-to-trial scatter. If it does not, the retest captures
    are not being found and every between-session verdict silently degrades
    to 'can\'t tell'."""
    wh = make_warehouse(n_athletes=24, sessions_per_athlete=3,
                        trials_per_session=14, retest_athletes=20,
                        session_drift_scale=0.5, seed=31)
    within = compute_reliability(wh.trials, ALL_TRIAL_METRICS)
    between = estimate_between_session(wh.trials, ALL_TRIAL_METRICS)
    assert not between.empty, "no retest pairs were found"
    merged = merge_reliability(within, between).set_index("metric")
    wi = within.set_index("metric")
    wider = sum(1 for m in between["metric"]
                if merged.loc[m, "sem"] > wi.loc[m, "sem"])
    assert wider >= 0.7 * len(between), (
        f"only {wider}/{len(between)} metrics had a wider between-session band")
    # And merge prefers the between-session number where it exists.
    assert (merged.loc[between["metric"], "source"] == SOURCE_BETWEEN).all()


def test_reliability_recovers_known_measurement_noise():
    """SEM should land close to the trial noise we actually injected."""
    true_sd = {m: 3.0 for m in ALL_TRIAL_METRICS}
    wh = make_warehouse(n_athletes=20, sessions_per_athlete=3,
                        trials_per_session=15, measurement_sd=true_sd, seed=4)
    rel = compute_reliability(wh.trials, ALL_TRIAL_METRICS)
    assert not rel.empty
    for _, r in rel.iterrows():
        assert 2.4 < r["sem"] < 3.6, (
            f"{r['metric']}: SEM {r['sem']:.2f}, expected ~3.0")
        # MDC is the 95% band for a single-trial comparison.
        assert r["mdc95_single"] == pytest.approx(
            CONFIG.mdc_multiplier * r["sem"], rel=1e-6)


def test_within_session_band_tightens_with_more_pitches():
    """Comparing two halves of the SAME bullpen is limited only by pitch-to-
    pitch scatter, so more pitches means a finer instrument."""
    rel = _rel(sem=4.0, source=SOURCE_WITHIN)
    single = rel.mdc_for_means("m", 1, 1, comparison=COMPARISON_WITHIN)
    many = rel.mdc_for_means("m", 20, 20, comparison=COMPARISON_WITHIN)
    assert many < single
    assert many == pytest.approx(single / np.sqrt(20), rel=1e-6)


def test_between_session_band_does_not_shrink_with_pitch_count():
    """Day-to-day error is shared by every pitch in a session, so throwing
    more of them cannot make you more certain the DAY was typical. Letting the
    band shrink here is what called ~60% of unchanged mechanics 'real' on data
    with realistic drift."""
    rel = _rel(sem=4.0, source=SOURCE_BETWEEN)
    few = rel.mdc_for_means("m", 5, 5)
    many = rel.mdc_for_means("m", 50, 50)
    assert few == pytest.approx(many)
    assert many == pytest.approx(CONFIG.mdc_z * 4.0 * np.sqrt(2))


def test_within_session_estimate_gives_a_provisional_band_not_a_shrunk_one():
    """Without repeat captures we still say something, but the band is the
    un-shrunk single-trial error and every verdict is flagged provisional.
    The shrunk version is what produced the false positives, so it must not
    appear for a between-session question at any pitch count."""
    rel = _rel(sem=4.0, source=SOURCE_WITHIN)
    band = rel.mdc_for_means("m", 12, 12)
    assert band == pytest.approx(CONFIG.mdc_z * 4.0 * np.sqrt(2))
    shrunk = rel.mdc_for_means("m", 12, 12, comparison=COMPARISON_WITHIN)
    assert band > shrunk * 3

    v = rel.classify("m", 50.0, n_before=12, n_after=12)
    assert v.verdict == VERDICT_REAL
    assert v.provisional
    assert "provisional" in v.explain()

    gaps = rel.reliability_gaps()
    assert gaps["needs_retest"] == ["m"]
    assert gaps["ready"] == []


def test_strict_mode_refuses_without_test_retest_data():
    """With the fallback off, a between-session question on a within-session
    estimate is answered honestly: we cannot tell."""
    strict = CONFIG.replace(allow_provisional_band=False)
    rel = ReliabilityTable(pd.DataFrame([{
        "metric": "m", "sem": 4.0, "cv_pct": None, "icc": 0.8,
        "mdc95_single": CONFIG.mdc_multiplier * 4.0, "sd_between": 8.0,
        "n_sessions": 40, "n_trials": 480, "n_athletes": 20,
        "source": SOURCE_WITHIN,
    }]), config=strict)
    assert rel.mdc_for_means("m", 12, 12) is None
    v = rel.classify("m", 50.0, n_before=12, n_after=12)
    assert v.verdict == VERDICT_UNKNOWN
    assert "twice within two weeks" in v.explain()


def test_real_verdicts_from_a_genuine_band_are_not_provisional():
    rel = _rel(sem=4.0, source=SOURCE_BETWEEN)
    v = rel.classify("m", 50.0, n_before=12, n_after=12)
    assert v.verdict == VERDICT_REAL
    assert not v.provisional
    assert "provisional" not in v.explain()


def test_no_verdict_text_calls_data_noise():
    """Every measurement is taken as fact. The vocabulary compares magnitudes;
    it never tells a coach a recorded difference was not real."""
    rel = _rel(sem=4.0)
    for delta in (0.2, 4.0, 40.0):
        txt = rel.classify("m", delta, n_before=6, n_after=6).explain().lower()
        for banned in ("noise", "unchanged", "not real", "spurious", "error band"):
            assert banned not in txt, f"{banned!r} appeared in: {txt}"
    from src.research.reliability import VERDICT_LABEL
    for label in VERDICT_LABEL.values():
        assert "noise" not in label.lower()


def test_verdict_false_positive_rate_is_calibrated():
    """The whole system rests on 'real' meaning something. Generate many
    comparisons where nothing changed and check the tiers fire at close to
    their advertised rates: ~5% real (95% band), ~20% real-or-likely (80%
    band). If this drifts, every green chip on every coach page is a lie.
    """
    rng = np.random.default_rng(101)
    sem, k = 5.0, 12
    rel = _rel(sem=sem)

    n_sim = 4000
    # Under the null, a between-session difference is distributed with
    # SD = SEM_between * sqrt(2). That is exactly what the band is built from,
    # which is what makes the tier rates predictable.
    se_of_diff = sem * np.sqrt(2.0)
    deltas = rng.normal(0.0, se_of_diff, size=n_sim)
    verdicts = [rel.classify("m", float(d), n_before=k, n_after=k).verdict
                for d in deltas]
    p_real = sum(v == VERDICT_REAL for v in verdicts) / n_sim
    p_flagged = sum(v != VERDICT_NOISE for v in verdicts) / n_sim

    assert 0.02 < p_real < 0.09, f"'real' fires at {p_real:.1%}, expected ~5%"
    assert 0.14 < p_flagged < 0.27, (
        f"'real or likely' fires at {p_flagged:.1%}, expected ~20%")


def test_verdict_has_power_for_a_true_change():
    """The complement: a change of two error bands must be caught nearly
    always, or the floor is set so high that nothing is ever reported."""
    sem, k = 5.0, 12
    rel = _rel(sem=sem)
    mdc = rel.mdc_for_means("m", k, k)
    rng = np.random.default_rng(7)
    se_of_diff = sem * np.sqrt(2.0)
    deltas = 2 * mdc + rng.normal(0.0, se_of_diff, size=2000)
    hits = sum(rel.classify("m", float(d), n_before=k, n_after=k).is_real
               for d in deltas)
    assert hits / 2000 > 0.95


def test_unknown_metric_is_unknown_not_real():
    rel = ReliabilityTable()
    v = rel.classify("never_seen", 99999.0, n_before=10, n_after=10)
    assert v.verdict == VERDICT_UNKNOWN
    assert not v.is_real
    assert "no repeat-capture baseline" in v.explain().lower()


def test_change_inside_band_is_called_unchanged():
    rel = _rel(sem=10.0)
    small = rel.classify("m", 0.5, n_before=10, n_after=10)
    assert small.verdict == VERDICT_NOISE
    # Descriptive, not a truth claim: it says how the move compares in size,
    # and never tells the coach the difference did not happen.
    txt = small.explain().lower()
    assert "usually varies" in txt
    assert "noise" not in txt
    assert "unchanged" not in txt
    # Band here is 1.96 * 10 * sqrt(2) = 27.7, so 40 clears it and 25 does not.
    assert rel.classify("m", 25.0, n_before=10, n_after=10).verdict != VERDICT_REAL
    big = rel.classify("m", 40.0, n_before=10, n_after=10)
    assert big.verdict == VERDICT_REAL


# ══════════════════════════════════════════════════════════════════════════
# Provenance gating
# ══════════════════════════════════════════════════════════════════════════

def test_carried_forward_assessment_is_excluded():
    """A profile value whose source session sits far outside the round window
    is a carry-forward. Differencing it against a fresh test invents change,
    so it must be dropped and reported, not silently included."""
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=2, seed=9)
    uuid = wh.uuid_for("Athlete 00")
    # Round 2's mobility value actually came from a year earlier.
    wh.snapshots[uuid][1]["source_dates"]["mobility"] = "2023-01-01"

    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        meta = rep.session_change["assessment_delta_meta"]
        stale = {e["metric"] for e in meta["excluded_stale"]}
        assert "mob_shoulder_ir" in stale
        deltas = rep.session_change["assessment_deltas"]
        if not deltas.empty:
            assert "mob_shoulder_ir" not in set(deltas["metric"])
        assert any("carried forward" in w for w in rep.warnings)


def test_same_source_session_is_not_a_change():
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=2, seed=13)
    uuid = wh.uuid_for("Athlete 01")
    same = wh.snapshots[uuid][0]["source_dates"]["athletic_screen_dj"]
    wh.snapshots[uuid][1]["source_dates"]["athletic_screen_dj"] = same

    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 01")
        meta = rep.session_change["assessment_delta_meta"]
        same_src = {e["metric"] for e in meta["excluded_same_source"]}
        assert "screen_dj_rsi" in same_src


def test_assessment_deltas_use_raw_not_z():
    """Raw values, because Z-scores carry norm drift."""
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=2, seed=17)
    uuid = wh.uuid_for("Athlete 02")
    s0, s1 = wh.snapshots[uuid][0], wh.snapshots[uuid][1]
    s0["raw_values"]["screen_cmj_jh_in"] = 20.0
    s1["raw_values"]["screen_cmj_jh_in"] = 24.0
    # Deliberately inconsistent Z-scores — if the code reads these, it fails.
    s0["z_scores"]["screen_cmj_jh_in"] = 0.0
    s1["z_scores"]["screen_cmj_jh_in"] = -5.0

    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 02")
        d = rep.session_change["assessment_deltas"]
        row = d[d["metric"] == "screen_cmj_jh_in"]
        assert not row.empty
        assert row.iloc[0]["delta"] == pytest.approx(4.0)
        assert row.iloc[0]["change"] == "improved"


# ══════════════════════════════════════════════════════════════════════════
# Cohort construction
# ══════════════════════════════════════════════════════════════════════════

def test_cohort_is_one_row_per_athlete():
    """An athlete with many sessions must not get many votes in the cohort
    distribution, which is what the old per-session aggregation did."""
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=4, seed=8)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        allm = rep.outliers["all_metrics"]
        assert not allm.empty
        # 12 athletes × 4 sessions = 48 rows under the old scheme; the cohort
        # excludes the athlete himself, so 11 is correct.
        assert set(allm["cohort_n"]) <= {11}, sorted(set(allm["cohort_n"]))


def test_athlete_excluded_from_his_own_cohort():
    wh = make_warehouse(n_athletes=9, sessions_per_athlete=2, seed=19)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 00")
        allm = rep.outliers["all_metrics"]
        assert (allm["cohort_n"] == 8).all()


def test_small_cohort_reports_rank_not_percentile():
    pos = percentile_or_rank(5.0, [1, 2, 3, 4, 6, 7], cohort_label="PRO")
    assert pos.mode == "rank"
    assert pos.percentile is None
    assert "highest of" in pos.describe("Braking rate")

    big = percentile_or_rank(5.0, list(range(40)), cohort_label="PRO")
    assert big.mode == "percentile"
    assert "percentile" in big.describe("Braking rate")

    tiny = percentile_or_rank(5.0, [1, 2], cohort_label="PRO")
    assert tiny.mode == "insufficient"
    assert "not enough" in tiny.describe("Braking rate").lower()


# ══════════════════════════════════════════════════════════════════════════
# Statistical helpers
# ══════════════════════════════════════════════════════════════════════════

def test_no_ci_below_bootstrap_floor():
    rng = np.random.default_rng(0)
    r = spearman_with_ci(rng.normal(size=4), rng.normal(size=4))
    assert r.ci_low is None and r.ci_high is None
    assert r.ci_spans_zero  # unknown is treated as 'could be zero'


def test_ci_excludes_zero_for_strong_relationship():
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = 2 * x + rng.normal(size=60) * 0.3
    r = spearman_with_ci(x, y)
    assert r.r > 0.9
    assert not r.ci_spans_zero


def test_max_r_null_scales_with_n_and_breadth():
    """More metrics scanned → a higher noise ceiling. More data → lower."""
    few = max_abs_r_null(n=12, n_metrics=5, n_sim=300)
    many = max_abs_r_null(n=12, n_metrics=200, n_sim=300)
    assert many.null_median > few.null_median

    small_n = max_abs_r_null(n=5, n_metrics=100, n_sim=300)
    big_n = max_abs_r_null(n=40, n_metrics=100, n_sim=300)
    assert small_n.null_median > big_n.null_median
    # The headline number from the review: at n=4 over ~150 metrics, noise
    # routinely produces a perfect-looking correlation.
    tiny = max_abs_r_null(n=4, n_metrics=150, n_sim=300)
    assert tiny.null_median > 0.95


def test_shrinkage_pulls_small_samples_harder():
    out = eb_shrink_means(means=[100.0, 100.0], counts=[3, 100],
                          within_sd=10.0, index=["few", "many"])
    grand = out.attrs["grand_mean"]
    if out.attrs["tau2"] > 0:
        assert out.loc["few", "weight"] < out.loc["many", "weight"]
    assert abs(out.loc["few", "shrunk"] - grand) <= abs(100.0 - grand) + 1e-9


def test_fisher_shrink_monotone_in_n():
    assert fisher_shrink_r(0.8, 4) < fisher_shrink_r(0.8, 20) < fisher_shrink_r(0.8, 200)
    assert fisher_shrink_r(0.8, 3) == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Metric dictionary
# ══════════════════════════════════════════════════════════════════════════

def test_curated_metrics_are_complete():
    """Every curated entry needs the fields the renderers rely on."""
    for key in CURATED_TRIAL_METRICS:
        d = DISPLAY.get(key)
        assert d.status == STATUS_CURATED, f"{key} is not curated"
        assert d.name and d.short and d.definition
        assert d.direction in ("higher_better", "lower_better", "context",
                               "load", "review")


def test_aliases_resolve():
    a = DISPLAY.get("pitch_force_lead_peak_braking_bw")
    b = DISPLAY.get("fm_lead_peak_braking_bw")
    assert a.name == b.name
    assert a.key != b.key  # the alias keeps its own key for traceability


def test_composer_handles_the_long_tail():
    """Uncurated Visual3D keys still come out readable — and marked as auto so
    they never reach a coach page unreviewed."""
    samples = [
        "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z",
        "kin_PROCESSED.Max_Elbow_Varus_Torque_Nm.X",
        "kin_TIMING.MaxShoulderVelTime.X",
        "kin_INCREMENT.Pelvis_Ang_Vel@MaxKneeHeight_800ms.Z",
        "fm_drive_peak_vertical_n",
    ]
    for s in samples:
        d = DISPLAY.get(s)
        assert d.status != STATUS_CURATED
        assert d.name and "None" not in d.name
        assert not d.is_coach_ready


def test_direction_language_respects_the_metric():
    # lower_better: a decrease is an improvement
    verb, _ = DISPLAY.describe_change("fm_lead_time_to_peak_fz_ms", -8.0)
    assert verb == "improved"
    # higher_better: a decrease is a decline
    verb, _ = DISPLAY.describe_change("fm_lead_peak_braking_bw", -2.0)
    assert verb == "declined"
    # load: never better or worse
    verb, _ = DISPLAY.describe_change("pitch_max_elbow_varus_torque_nm", 6.0)
    assert verb == "increased"
    assert DISPLAY.signed_toward_better("pitch_max_elbow_varus_torque_nm", 6.0) is None


# ══════════════════════════════════════════════════════════════════════════
# Config & caching
# ══════════════════════════════════════════════════════════════════════════

def test_config_fingerprint_changes_with_thresholds():
    a = CONFIG.fingerprint()
    b = CONFIG.replace(min_trials_within_session=99).fingerprint()
    assert a != b
    assert CONFIG.fingerprint() == a  # frozen: replace() did not mutate


def test_config_rejects_unknown_field():
    with pytest.raises(ValueError):
        CONFIG.replace(not_a_real_setting=1)


def test_deep_dive_hits_the_warehouse_once():
    """The old run issued ~14 full trial pulls per athlete."""
    from src.research import loaders
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=2, seed=2)
    with patched_warehouse(wh):
        loaders.clear_cache()
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        run_athlete_deep_dive("Athlete 00")
        stats = loaders.cache_stats()
        assert stats["queries"] <= 2, stats
        assert stats["hits"] >= stats["misses"]


# ══════════════════════════════════════════════════════════════════════════
# Rendering smoke tests
# ══════════════════════════════════════════════════════════════════════════

def test_coach_report_renders(tmp_path):
    wh = make_warehouse(n_athletes=14, sessions_per_athlete=3,
                        trials_per_session=12,
                        planted=[PlantedEffect("fm_lead_peak_braking_bw",
                                               within_athlete_beta=0.2)],
                        seed=44)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        from src.research.coach_report import render_coach_report
        rep = run_athlete_deep_dive("Athlete 05")
        path = render_coach_report(rep, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "Athlete 05" in html
        # No raw warehouse keys in the coach-facing headline cards.
        head_section = html.split("The evidence")[0]
        assert "fm_lead_" not in head_section
        assert "kin_PROCESSED" not in head_section


def test_roster_renders(tmp_path):
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=3, seed=55)
    with patched_warehouse(wh):
        from src.research.roster import build_roster, render_roster_report
        r = build_roster()
        assert len(r.rows) == 10
        path = render_roster_report(r, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "Squad overview" in html
        assert "Athlete 00" in html


def test_report_with_no_data_does_not_crash(tmp_path):
    """An athlete with one session and nothing else still produces a page that
    says so, rather than raising."""
    wh = make_warehouse(n_athletes=6, sessions_per_athlete=1,
                        trials_per_session=5, seed=66)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        from src.research.coach_report import render_coach_report
        rep = run_athlete_deep_dive("Athlete 00")
        assert rep.session_change.get("skipped")
        path = render_coach_report(rep, output_dir=tmp_path)
        assert path.exists()


# ══════════════════════════════════════════════════════════════════════════
# Metric hygiene — what belongs in a change table
# ══════════════════════════════════════════════════════════════════════════

def test_model_constants_are_excluded_from_change():
    """Segment lengths led Ryan Chasse's change table with a ratio of 7.79e+14,
    because they are near-constant within a session (so their spread is ~0) and
    they move between sessions when a marker is placed differently. They are a
    property of the rig, not the pitcher."""
    from src.research.metric_filters import exclude_from_change, filter_for_change

    for m in ("kin_MODEL.RTA_Seg_Length.X", "kin_MODEL.LTH_Seg_Length.X",
              "kin_PROCESSED.Frame_rate.X", "fm_lead_plate_id"):
        why = exclude_from_change(m)
        assert why, f"{m} should be excluded"
        assert "marker" in why or "index" in why or "identifier" in why

    kept, dropped = filter_for_change([
        "kin_MODEL.RTA_Seg_Length.X",
        "fm_lead_peak_braking_bw",
        "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z",
    ])
    assert "kin_MODEL.RTA_Seg_Length.X" not in kept
    assert "fm_lead_peak_braking_bw" in kept
    assert "kin_PROCESSED.Pelvis_Ang_Vel@Release.Z" in kept
    assert len(dropped) == 1


def test_bodyweight_kept_once_in_kg():
    """Weight matters, but not twice in two units."""
    from src.research.metric_filters import filter_for_change
    kept, dropped = filter_for_change(["fm_body_weight_kg", "fm_body_weight_n"])
    assert kept == ["fm_body_weight_kg"]
    assert "fm_body_weight_n" in dropped
    assert "fm_body_weight_kg" in dropped["fm_body_weight_n"]


def test_kinetics_and_kinematics_separate():
    """Force metrics outnumber angle metrics, so one combined table always
    buries the kinematics. They get their own."""
    from src.research.metric_filters import (KINEMATICS, KINETICS, TIMING,
                                             classify_family, split_by_family)
    assert classify_family("fm_lead_peak_braking_bw") == KINETICS
    assert classify_family("pitch_lead_leg_grf_mag_max") == KINETICS
    assert classify_family("pitch_max_elbow_varus_torque_nm") == KINETICS
    assert classify_family("pitch_hip_shoulder_sep_at_fc") == KINEMATICS
    assert classify_family("kin_PROCESSED.Pelvis_Ang_Vel@Release.Z") == KINEMATICS
    assert classify_family("fm_fc_to_br_duration_s") == TIMING
    assert classify_family("velocity_mph") == "output"

    groups = split_by_family(["fm_lead_peak_braking_bw",
                              "pitch_hip_shoulder_sep_at_fc"])
    assert set(groups) == {KINETICS, KINEMATICS}


# ══════════════════════════════════════════════════════════════════════════
# Unit / scale discontinuities
# ══════════════════════════════════════════════════════════════════════════

def _grf_unit_change_frame() -> pd.DataFrame:
    """Reproduces the real Chasse pattern: every GRF metric ~1000x smaller in
    the later session, everything else unchanged."""
    rows = []
    early = {"fm_lead_peak_vertical_bw": 2110.0, "fm_lead_peak_braking_bw": 1981.0,
             "fm_drive_peak_vertical_bw": 729.0, "fm_lead_impulse_v_into_ball_bws": 170.0}
    stable = {"fm_fc_to_br_duration_s": 0.145}
    rng = np.random.default_rng(3)
    for sess, scale in ((pd.Timestamp("2024-09-10").date(), 1.0),
                        (pd.Timestamp("2026-09-15").date(), 1 / 1082.0)):
        for t in range(6):
            rec = {"athlete_uuid": "a1", "name": "Test", "session_date": sess,
                   "trial_index": t, "velocity_mph": 90 + rng.normal(0, 1)}
            for k, v in early.items():
                rec[k] = v * scale * (1 + rng.normal(0, 0.02))
            for k, v in stable.items():
                rec[k] = v * (1 + rng.normal(0, 0.02))
            rows.append(rec)
    return pd.DataFrame(rows)


def test_unit_change_is_detected_not_reported_as_change():
    from src.research.unit_guard import audit_scale_changes
    df = _grf_unit_change_frame()
    metrics = [c for c in df.columns if c.startswith("fm_")]
    audit = audit_scale_changes(df, metrics)

    assert audit.has_findings
    assert audit.systemic, "four metrics jumping by the same factor is systemic"
    s = audit.systemic[0]
    assert 900 < s.median_ratio < 1300
    assert len(s.metrics) >= 3
    assert "fm_fc_to_br_duration_s" not in s.metrics  # unchanged one is clean
    text = s.describe()
    assert "pipeline" in text
    assert "not a change in the athlete" in text


def test_scale_break_blocks_the_comparison():
    from src.research.unit_guard import audit_scale_changes
    df = _grf_unit_change_frame()
    metrics = [c for c in df.columns if c.startswith("fm_")]
    audit = audit_scale_changes(df, metrics)
    d0 = pd.Timestamp("2024-09-10").date()
    d1 = pd.Timestamp("2026-09-15").date()
    blocked = audit.blocked_metrics_between(d0, d1)
    assert "fm_lead_peak_vertical_bw" in blocked
    assert "fm_fc_to_br_duration_s" not in blocked


def test_normal_data_triggers_no_scale_warning():
    """The guard must not fire on ordinary training change."""
    from src.research.unit_guard import audit_scale_changes
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=3,
                        trials_per_session=10, seed=77)
    audit = audit_scale_changes(wh.trials, ALL_TRIAL_METRICS)
    assert not audit.systemic, [s.describe() for s in audit.systemic]


def test_table_survives_duplicate_column_labels():
    """Two metrics can share a short display name. A duplicate header turns
    d[c] into a DataFrame and blows up on .dtype, so the renderer dedupes."""
    from src.research import render_kit as rk
    df = pd.DataFrame([[1, 2.5, 3], [4, 5.5, 6]], columns=["Pitch", "Peak", "Peak"])
    html = rk.table(df)
    assert "<table>" in html
    # Two header cells, not three — the duplicate is dropped, not crashed on.
    # (Counting "<th " avoids also matching "<thead>".)
    assert html.count("<th ") == 2


def test_per_pitch_table_disambiguates_labels(tmp_path):
    """End to end: the coach report renders even when two of the moving
    metrics share a short name."""
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=3,
                        trials_per_session=6, retest_athletes=10, seed=5)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        from src.research.coach_report import render_coach_report
        rep = run_athlete_deep_dive("Athlete 04")
        path = render_coach_report(rep, output_dir=tmp_path)
        html = path.read_text(encoding="utf-8")
        assert "Every pitch" in html


# ══════════════════════════════════════════════════════════════════════════
# Session matrix — one shape, latest-vs-previous
# ══════════════════════════════════════════════════════════════════════════

def test_matrix_compares_last_two_not_first_to_last():
    """The headline change must be the most recent move, not a delta
    accumulated across every capture in between. A first-to-last difference
    hides everything that happened in the middle and inherits every bit of
    error along the way."""
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=4,
                        trials_per_session=10, seed=91)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 03")
        sm = rep.session_change["matrix"]

        assert sm.latest == sm.session_dates[-1]
        assert sm.previous == sm.session_dates[-2]
        # Every capture stays on the page as its own column.
        assert len(sm.value_columns) == len(sm.session_dates) >= 3

        row = sm.frame[sm.frame["change"].notna()].iloc[0]
        expected = row[f"s_{sm.latest}"] - row[f"s_{sm.previous}"]
        assert row["change"] == pytest.approx(expected, abs=1e-6)
        # And the first capture is present but not what the change measures.
        first = row[f"s_{sm.session_dates[0]}"]
        assert first is not None


def test_assessment_history_is_not_gated_on_3d_captures(tmp_path):
    """Assessments render on their own test dates, through the same matrix as
    the movement tables. Anchoring them to 3D captures hid most of a real
    athlete's testing: 29 of his 39 test dates fell outside every capture
    window, and a year of screens taken after his last capture could not
    appear at any window width."""
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=4,
                        trials_per_session=10, retest_athletes=10, seed=92)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        from src.research.coach_report import render_coach_report
        from src.research.session_matrix import ASSESSMENT_LABEL
        rep = run_athlete_deep_dive("Athlete 03")
        history = rep.session_change["assessment_history"]
        if not history:
            pytest.skip("no assessment values in this synthetic draw")

        for mod, sm in history.items():
            # Its own dates, and the canonical column shape.
            assert list(sm.value_columns) == [f"s_{d}" for d in sm.session_dates]
            assert sm.latest == sm.session_dates[-1]
            # A 3D-derived metric in the assessment table is the bug that made
            # pitching kinematics show up as screen results.
            assert not sm.frame["metric"].str.startswith(("pitch_", "hit_")).any()

        html = render_coach_report(rep, output_dir=tmp_path).read_text("utf-8")
        assert "Assessment history" in html
        label = ASSESSMENT_LABEL.get(next(iter(history)))
        if label:
            assert label in html


def test_coverage_names_dates_the_profiler_has_never_seen(tmp_path):
    """A test date with no profile row cannot appear anywhere in the report,
    however wide the round window is. That is a different problem from being
    outside a window, and the report has to say which one it is."""
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=3,
                        trials_per_session=8, retest_athletes=8, seed=93)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 02")
        cov = rep.assessment_coverage or {}
        assert "unprofiled_dates" in cov
        assert isinstance(cov["unprofiled_dates"], list)


def test_constant_metric_does_not_produce_an_absurd_ratio():
    """Body weight is one number repeated on every pitch, so its within-session
    spread is float dust. Dividing by it produced 391,482,406,602,053x usual in
    a real report. A band has to be a real fraction of the quantity."""
    from src.research.reliability import ChangeVerdict, VERDICT_BEYOND
    from src.research.session_matrix import _usable_ratio
    dust = ChangeVerdict("fm_body_weight_kg", -4.54, 1.16e-14, 3.9e14,
                         VERDICT_BEYOND, 5, 4, "within_session")
    assert not _usable_ratio(dust, 120.2, 115.7)
    real = ChangeVerdict("fm_lead_peak_braking_bw", 0.4, 0.2, 2.0,
                         VERDICT_BEYOND, 5, 4, "within_session")
    assert _usable_ratio(real, 1.8, 2.2)


def test_matrix_reports_captures_that_are_missing_their_metrics():
    """A capture written without its full metric set shows as a column of
    dashes. The report has to name it, or the reader cannot tell a processing
    gap from an athlete who was not measured."""
    wh = make_warehouse(n_athletes=10, sessions_per_athlete=3,
                        trials_per_session=8, seed=101)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 01")
        sm = rep.session_change["matrix"]
        assert sm.coverage and set(sm.coverage) == set(sm.session_dates)
        # Nothing is thin in a clean synthetic warehouse.
        assert sm.thin_captures() == []
        # Knock one capture's values out and it must be flagged.
        thin_date = sm.session_dates[0]
        sm.coverage[thin_date] = 1
        assert [d for d, _, _ in sm.thin_captures()] == [thin_date]


def test_coverage_counts_every_test_date_not_one_per_round():
    """A round keeps the assessment date CLOSEST to its anchor. Counting from
    that kept date reported 11 Proteus sessions for an athlete with 41 — the
    rest were inside a window but not named by it, so they were tallied
    neither as paired nor as outside and vanished."""
    from datetime import date
    from types import SimpleNamespace
    from src.research.athlete_deep_dive import section_assessment_coverage
    from src.research.assessment_rounds import AssessmentRound

    anchor = date(2025, 10, 24)
    proteus = [date(2025, 10, 3), date(2025, 10, 11), date(2025, 10, 24),
               date(2025, 11, 4), date(2026, 2, 10)]   # last one far outside
    rnd = AssessmentRound(
        round_id="R1", anchor_type="pitching_3d", anchor_date=anchor,
        window_start=date(2025, 8, 25), window_end=date(2025, 12, 23),
        sources_present={"proteus": anchor}, profile_id=1,
        profile_as_of_date=anchor)
    ctx = SimpleNamespace(
        rounds=[rnd],
        rounds_meta={
            "window_days": 60,
            "all_source_dates": {"proteus": proteus},
            "dates_in_a_window": {"proteus": proteus[:4]},
            "orphaned_assessments": {"proteus": [proteus[-1]]},
        },
        snapshots={})

    cov = section_assessment_coverage(ctx)
    row = cov["by_source"].iloc[0]
    assert row["total_dates"] == 5          # not 2 (the kept one + the orphan)
    assert row["in_a_round"] == 4
    assert row["outside_every_round"] == 1
    assert row["latest_used"] == str(anchor)


def test_dust_band_never_becomes_a_verdict():
    """The same guard at the source, so the appendix and any other consumer of
    classify() gets it too — not just the matrix."""
    from src.research.reliability import ReliabilityTable
    import pandas as pd
    tbl = ReliabilityTable(pd.DataFrame([{
        "metric": "fm_body_weight_kg", "sem": 1e-15, "icc": 0.99,
        "mdc95_single": 1e-15, "n_athletes": 5, "n_sessions": 5,
        "n_trials": 20, "source": "within_session",
    }]))
    cv = tbl.classify("fm_body_weight_kg", 6.8)
    assert cv.ratio is None
    assert not cv.exceeds_typical


def test_wrapped_scalars_are_read_as_numbers():
    """Two pipeline versions wrote the metrics blob: one with
    jsonlite::toJSON(auto_unbox = TRUE) storing 1.23, an older one storing the
    R vector it was handed — [1.23]. A real athlete's 2025 captures came back
    7,710 wrapped values and 0 numbers, which read as two sessions with no
    kinematics. Same measurement either way."""
    from src.research.pitching_deep import _coerce_float
    assert _coerce_float(1.23) == 1.23
    assert _coerce_float([1.23]) == 1.23
    assert _coerce_float([[1.23]]) == 1.23
    assert _coerce_float({"value": 1.23}) == 1.23
    # Genuinely multi-valued or non-numeric stays rejected.
    assert _coerce_float([1, 2]) is None
    assert _coerce_float("abc") is None
    assert _coerce_float(True) is None
    assert _coerce_float(None) is None


def test_discontinued_screens_are_not_round_sources():
    """NMT was retired in 2024. Keeping it as a source put a dead row in every
    coverage table and an unprofiled-date warning on a test nobody will take."""
    from src.research.assessment_rounds import _SOURCE_TABLES
    assert "screen_nmt" not in _SOURCE_TABLES
    for keep in ("screen_cmj", "screen_dj", "screen_slv", "screen_ppu"):
        assert keep in _SOURCE_TABLES


def _angle_through_zero_frame() -> pd.DataFrame:
    """An angle that happens to sit near zero at one capture and returns to
    its usual magnitude afterwards. The raw numbers look ordinary to anyone
    reading the table, which is why flagging it reads as a bug."""
    rows = []
    rng = np.random.default_rng(11)
    series = [(pd.Timestamp("2024-09-10").date(), 5.0),
              (pd.Timestamp("2025-01-27").date(), 0.05),
              (pd.Timestamp("2025-10-24").date(), 4.6),
              (pd.Timestamp("2026-09-15").date(), 5.2)]
    for sess, v in series:
        for t in range(6):
            rows.append({"athlete_uuid": "a1", "name": "Test",
                         "session_date": sess, "trial_index": t,
                         "velocity_mph": 90 + rng.normal(0, 1),
                         "kin_PROCESSED.Trunk_Angle@Setup.Y":
                             v * (1 + rng.normal(0, 0.02))})
    return pd.DataFrame(rows)


def test_one_capture_excursion_is_not_a_unit_change():
    """A unit change is permanent — everything after the boundary is on the
    new scale. A value that dips and comes back is arithmetic."""
    from src.research.unit_guard import audit_scale_changes
    df = _angle_through_zero_frame()
    audit = audit_scale_changes(df, ["kin_PROCESSED.Trunk_Angle@Setup.Y"])
    assert not audit.systemic
    assert not audit.breaks


def test_only_systemic_breaks_withhold_a_comparison():
    """One metric moving 20x alone is usually a small denominator. Blocking on
    it printed 'units changed' against 2,136 N -> 2,515 N in a real report."""
    from src.research.unit_guard import (audit_scale_changes, ScaleBreak,
                                         UnitAudit)
    d0 = pd.Timestamp("2025-10-24").date()
    d1 = pd.Timestamp("2026-09-15").date()
    audit = UnitAudit(breaks=[ScaleBreak(
        metric="fm_lead_peak_vertical_n", from_date=d0, to_date=d1,
        from_value=1.0, to_value=30.0, ratio=30.0, direction="grew")])
    assert audit.blocked_details_between(d0, d1) == {}
    assert [b.metric for b in audit.isolated_breaks_between(d0, d1)] \
        == ["fm_lead_peak_vertical_n"]

    # The real GRF pattern still blocks, and says by how much.
    real = audit_scale_changes(
        _grf_unit_change_frame(),
        [c for c in _grf_unit_change_frame().columns if c.startswith("fm_")])
    det = real.blocked_details_between(pd.Timestamp("2024-09-10").date(),
                                       pd.Timestamp("2026-09-15").date())
    assert "fm_lead_peak_vertical_bw" in det
    assert "x" in det["fm_lead_peak_vertical_bw"]


def test_absolute_capture_clock_times_are_not_differenced():
    """`fm_fc_time_s` is where an event fell on a clock that started when
    somebody hit record. A real report differenced these and announced
    "+0.409 s" on a peak-force timing metric — that is the recording operator,
    not the pitcher. Intervals survive; timestamps do not."""
    from src.research.metric_filters import filter_for_change
    kept, dropped = filter_for_change([
        "fm_fc_time_s", "fm_br_time_s", "fm_lead_peak_vertical_time_s",
        "fm_axis_vertical",
        "fm_fc_to_br_duration_s", "fm_foot_loading_time_ms",
        "fm_peak_v_to_peak_b_lag_ms",
    ])
    assert kept == ["fm_fc_to_br_duration_s", "fm_foot_loading_time_ms",
                    "fm_peak_v_to_peak_b_lag_ms"]
    assert set(dropped) == {"fm_fc_time_s", "fm_br_time_s",
                            "fm_lead_peak_vertical_time_s", "fm_axis_vertical"}


def test_every_metric_the_coverage_report_flagged_now_has_a_name():
    """The 97 keys `research metric-coverage --status raw` listed against real
    data. Two segment codes (RPV / RTA centre-of-mass speeds) are deliberately
    still raw — nobody has confirmed which segment they are."""
    from src.research.metric_display import DISPLAY
    for key in ["score", "kin_BALLSPEED.SPIN_RATE.X",
                "kin_STRIDE_LENGTH.STRIDE_LENGTH.X",
                "kin_MODEL.LTH_Seg_Length.X",
                "fm_body_weight_kg", "fm_mer_time_s",
                "fm_virtual_plate_xcheck_pct", "fm_axis_braking",
                "screen_slv_right_time_to_rpd_max_s",
                "proteus_hitter_shotput_power_high",
                "mob_cervical_rotation",
                "hit_lead_leg_block_delta", "hit_trunk_total_rotation_mean"]:
        assert DISPLAY.is_coach_ready(key), key
    assert not DISPLAY.is_coach_ready("hit_max_rpv_cgpos_linear_vel")


def test_body_weight_is_not_a_cohort_finding(tmp_path):
    """Curating body weight promoted it onto the coach page as 'Body weight:
    100th percentile' — twice, once in kilograms and once in newtons. He is
    the biggest guy on file; that is not a finding about his delivery, and two
    units is not two findings."""
    wh = make_warehouse(n_athletes=14, sessions_per_athlete=3,
                        trials_per_session=10, seed=202)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 06")
        allm = rep.outliers.get("all_metrics")
        if allm is None or allm.empty:
            pytest.skip("no cohort positions in this draw")
        keys = set(allm["metric"])
        assert "fm_body_weight_kg" not in keys
        assert "fm_body_weight_n" not in keys
        for f in rep.headlines:
            assert "body_weight" not in (f.metric or "")


def test_velocity_headline_matches_the_captures_it_names():
    """The headline said "down 1.9 mph since 2025-10-24" for an athlete who
    threw 91.9 at that capture and 87.5 at the next — the real move was 4.4.
    Velocity was missing from the matrix, so the finding fell back to the
    first-to-last delta while keeping the latest-vs-previous label."""
    wh = make_warehouse(n_athletes=12, sessions_per_athlete=4,
                        trials_per_session=10, seed=77)
    with patched_warehouse(wh):
        from src.research.athlete_deep_dive import run_athlete_deep_dive
        rep = run_athlete_deep_dive("Athlete 02")
        sm = rep.session_change["matrix"]
        vrow = sm.frame[sm.frame["metric"] == "velocity_mph"]
        assert not vrow.empty, "velocity belongs in the table it explains"

        r = vrow.iloc[0]
        expected = r[f"s_{sm.latest}"] - r[f"s_{sm.previous}"]
        assert r["change"] == pytest.approx(expected, abs=1e-6)

        velo = [f for f in rep.headlines if f.metric == "velocity_mph"]
        if velo:
            head = velo[0].headline
            if str(sm.previous) in head:
                # The number quoted must be the one for those two captures.
                import re as _re
                m = _re.search(r"([\d.]+) mph", head)
                assert m and abs(float(m.group(1)) - abs(expected)) < 0.15, head
