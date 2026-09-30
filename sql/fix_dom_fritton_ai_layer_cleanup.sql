-- Cleans up ai_layer.* derived data for Dom Fritton (54d697ac-d3d6-4ba1-a5ca-d1a6b55f76a6)
-- generated before the f_mobility misattribution bug was fixed in OctaneBiomechBackend
-- (see that repo's uais/sql/fix_mobility_misattribution*.sql). This app doesn't snapshot
-- f_mobility -- profile/coach-report queries it live -- but it DOES persist derived rows
-- in ai_layer.athlete_profiles and ai_layer.research_findings, which are now stale.
--
-- Verified via ai_layer.athlete_profiles.source_dates->>'mobility': 18 profile rows were
-- built entirely to capture a bogus mobility test date (2026-05-06 through 2026-09-15) that
-- has since been deleted from f_mobility. Nothing else changed on any of those dates --
-- pitching/proteus/screen fields are all just carried forward from earlier real sessions --
-- so the row itself has no reason to exist once the bogus mobility date is gone.
-- His real dates (2025-09-25, and today's 2026-09-22 from the profile you just ran) are
-- untouched.
BEGIN;

DELETE FROM ai_layer.athlete_profiles
WHERE athlete_uuid = '54d697ac-d3d6-4ba1-a5ca-d1a6b55f76a6'
  AND (source_dates->>'mobility')::date IN (
    '2026-05-06','2026-05-15','2026-05-27','2026-05-28','2026-06-08','2026-06-11',
    '2026-06-15','2026-06-16','2026-07-07','2026-07-08','2026-07-09','2026-07-10',
    '2026-08-03','2026-08-07','2026-08-18','2026-09-07','2026-09-10','2026-09-15'
  );

-- Both coach-report runs from today (13:57 and 14:02) were generated against the polluted
-- f_mobility data (6 mobility dates instead of his real 1) -- delete the saved findings so
-- --open-previous / v_latest_research_findings don't resurface the stale report.
DELETE FROM ai_layer.research_findings
WHERE athlete_uuid = '54d697ac-d3d6-4ba1-a5ca-d1a6b55f76a6'
  AND report_type = 'athlete_deep_dive'
  AND generated_at::date = '2026-09-23';

COMMIT;

-- After running this, also delete the two stale HTML files (not SQL, just files):
--   outputs\coach_report_Dom_Fritton_20260923_1357.html
--   outputs\coach_report_Dom_Fritton_20260923_1402.html
-- Then re-run:
--   python -m src.main research coach-report "Dom Fritton"
-- It will recompute from the now-clean f_mobility and should show only his real
-- 2025-09-25 and 2026-09-22 mobility sessions.
