-- ═══════════════════════════════════════════════════════════════════════
-- Research layer persistence
--
-- Before this, a deep dive wrote a timestamped HTML file and nothing else.
-- Consequences:
--   * no way to answer "what did we tell this coach in June"
--   * no way to diff a report against its predecessor
--   * cohort-level statistics were recomputed inside every per-athlete
--     report, so two reports run a day apart could disagree with no record
--     of why
--   * a Z-score could not be traced to the norms that produced it
--
-- Idempotent: safe to run repeatedly.
-- ═══════════════════════════════════════════════════════════════════════

CREATE SCHEMA IF NOT EXISTS ai_layer;


-- ───────────────────────────────────────────────────────────────────────
-- 1. Norm versioning
--
-- `refresh_norms` does DELETE + INSERT per metric with no version stamp, so
-- a profile built in November 2024 and one built in October 2025 were scored
-- against different norm tables. Differencing their Z-scores mixed athlete
-- change with population drift and nothing recorded that it had happened.
--
-- Stamping a version on the norms and on every profile makes the mixing
-- visible: if two profiles carry different norms_version, their Z-delta is
-- not a clean measurement and the analysis can say so (or fall back to raw
-- values, which is what the deep dive now does by default).
-- ───────────────────────────────────────────────────────────────────────

ALTER TABLE ai_layer.assessment_norms
    ADD COLUMN IF NOT EXISTS norms_version TEXT;

ALTER TABLE ai_layer.assessment_norms
    ADD COLUMN IF NOT EXISTS computed_at TIMESTAMPTZ DEFAULT now();

ALTER TABLE ai_layer.athlete_profiles
    ADD COLUMN IF NOT EXISTS norms_version TEXT;

CREATE INDEX IF NOT EXISTS idx_athlete_profiles_norms_version
    ON ai_layer.athlete_profiles (norms_version);


-- ───────────────────────────────────────────────────────────────────────
-- 2. Measurement reliability
--
-- One row per (metric, stratum): how much this measurement moves when
-- nothing changed. Everything downstream that says "this is a real change"
-- resolves through here.
-- ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ai_layer.metric_reliability (
    id                BIGSERIAL PRIMARY KEY,
    metric            TEXT        NOT NULL,
    stratum           TEXT        NOT NULL DEFAULT 'ALL',
    sem               NUMERIC     NOT NULL,   -- typical error of one trial
    cv_pct            NUMERIC,                -- sem as % of the mean
    icc               NUMERIC,                -- between-athlete share of variance
    mdc95_single      NUMERIC     NOT NULL,   -- 1.96*sqrt(2)*sem
    sd_between        NUMERIC,
    n_sessions        INTEGER     NOT NULL,
    n_trials          INTEGER     NOT NULL,
    n_athletes        INTEGER     NOT NULL,
    source            TEXT        NOT NULL,   -- within_session | between_session
    config_hash       TEXT,
    analysis_version  TEXT,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (metric, stratum)
);

COMMENT ON TABLE ai_layer.metric_reliability IS
  'Measurement-noise floor per metric. source=within_session is a LOWER BOUND '
  '(trial scatter only); between_session additionally carries marker placement, '
  'calibration and day-to-day biology and is preferred where available.';

CREATE INDEX IF NOT EXISTS idx_metric_reliability_metric
    ON ai_layer.metric_reliability (metric);


-- ───────────────────────────────────────────────────────────────────────
-- 3. Cohort findings
--
-- Group-level statistics, computed once by a scheduled job instead of being
-- recomputed inside every per-athlete report. The stratum velocity-correlate
-- table is identical for every PRO pitcher; recomputing it per athlete burned
-- time AND let two reports disagree.
-- ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ai_layer.cohort_findings (
    id                BIGSERIAL PRIMARY KEY,
    analysis          TEXT        NOT NULL,   -- velocity_within_athlete | kin_to_assessment | ...
    stratum           TEXT        NOT NULL,   -- 'PRO' | 'COLLEGE' | ... | 'ALL'
    target            TEXT        NOT NULL,   -- what it was correlated against
    metric            TEXT        NOT NULL,
    r                 NUMERIC,
    ci_low            NUMERIC,
    ci_high           NUMERIC,
    p_value           NUMERIC,
    q_value           NUMERIC,
    fdr_significant   BOOLEAN,
    n                 INTEGER,
    n_athletes        INTEGER,
    tier              TEXT,                   -- strong | suggestive | insufficient
    extra             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    config_hash       TEXT,
    analysis_version  TEXT,
    computed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (analysis, stratum, target, metric)
);

CREATE INDEX IF NOT EXISTS idx_cohort_findings_lookup
    ON ai_layer.cohort_findings (analysis, stratum, target);
CREATE INDEX IF NOT EXISTS idx_cohort_findings_sig
    ON ai_layer.cohort_findings (fdr_significant) WHERE fdr_significant;


-- ───────────────────────────────────────────────────────────────────────
-- 4. Per-athlete research findings
--
-- The structured payload behind a rendered report. The HTML becomes a VIEW
-- over this rather than the finding itself, which is what makes "what changed
-- since last month's report" answerable.
-- ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ai_layer.research_findings (
    id                  BIGSERIAL PRIMARY KEY,
    athlete_uuid        UUID        NOT NULL,
    report_type         TEXT        NOT NULL DEFAULT 'athlete_deep_dive',
    focus_session_date  DATE,
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    config_hash         TEXT,
    analysis_version    TEXT,
    n_findings          INTEGER     NOT NULL DEFAULT 0,
    velocity_delta      NUMERIC,
    velocity_verdict    TEXT,
    findings            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    summary             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    warnings            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    output_paths        JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_research_findings_athlete
    ON ai_layer.research_findings (athlete_uuid, generated_at DESC);
CREATE INDEX IF NOT EXISTS idx_research_findings_type
    ON ai_layer.research_findings (report_type, generated_at DESC);
-- Findings are queried by metric often enough to be worth the GIN index.
CREATE INDEX IF NOT EXISTS idx_research_findings_gin
    ON ai_layer.research_findings USING GIN (findings jsonb_path_ops);


-- ───────────────────────────────────────────────────────────────────────
-- 5. Convenience view — the current state of every athlete, for the roster
-- ───────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW ai_layer.v_latest_research_findings AS
SELECT DISTINCT ON (athlete_uuid, report_type)
       athlete_uuid, report_type, focus_session_date, generated_at,
       config_hash, analysis_version, n_findings,
       velocity_delta, velocity_verdict, findings, summary, output_paths
FROM   ai_layer.research_findings
ORDER  BY athlete_uuid, report_type, generated_at DESC;
