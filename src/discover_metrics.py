"""
Discovery helper for the deficit profiler.

Long-form kinematics tables store the actual metric strings in `metric_name`,
and Proteus stores exercise variants in `exercise_name`/`movement`. Before we
write the profiler we need to know the *exact* strings present in your data.

This script prints, for each modality, the distinct metric_name / exercise_name /
column values available, along with row counts and (where useful) frame ranges.

Run:
    python -m src.discover_metrics              # everything
    python -m src.discover_metrics pitching     # one mode
    python -m src.discover_metrics pitching hitting proteus
"""
from __future__ import annotations

import sys

from src.db import backend_conn, query


def _print_table(title: str, rows: list[dict], cols: list[str] | None = None,
                 limit: int = 100) -> None:
    print(f"\n=== {title} ({len(rows)} rows) ===")
    if not rows:
        print("(no rows)")
        return
    if cols is None:
        cols = list(rows[0].keys())
    sample = rows[:limit]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in sample)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in sample:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if len(rows) > limit:
        print(f"... and {len(rows) - limit} more")


def pitching() -> None:
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT metric_name,
                   COUNT(*) AS n_rows,
                   COUNT(DISTINCT athlete_uuid) AS n_athletes,
                   MIN(frame) AS min_frame,
                   MAX(frame) AS max_frame
            FROM public.f_kinematics_pitching
            GROUP BY metric_name
            ORDER BY n_rows DESC
        """)
        _print_table("f_kinematics_pitching: distinct metric_name", rows,
                     cols=["metric_name", "n_rows", "n_athletes", "min_frame", "max_frame"])

        trials_cols = query(conn, """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'f_pitching_trials'
            ORDER BY ordinal_position
        """)
        _print_table("f_pitching_trials columns", trials_cols,
                     cols=["column_name", "data_type"], limit=200)


def hitting() -> None:
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT metric_name,
                   COUNT(*) AS n_rows,
                   COUNT(DISTINCT athlete_uuid) AS n_athletes,
                   MIN(frame) AS min_frame,
                   MAX(frame) AS max_frame
            FROM public.f_kinematics_hitting
            GROUP BY metric_name
            ORDER BY n_rows DESC
        """)
        _print_table("f_kinematics_hitting: distinct metric_name", rows,
                     cols=["metric_name", "n_rows", "n_athletes", "min_frame", "max_frame"])

        trials_cols = query(conn, """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'f_hitting_trials'
            ORDER BY ordinal_position
        """)
        _print_table("f_hitting_trials columns", trials_cols,
                     cols=["column_name", "data_type"], limit=200)


def mobility() -> None:
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'f_mobility'
              AND data_type IN ('numeric', 'integer', 'double precision')
            ORDER BY column_name
        """)
        _print_table("f_mobility numeric columns", rows,
                     cols=["column_name", "data_type"], limit=300)


def proteus() -> None:
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT exercise_name, movement, dominance,
                   COUNT(*) AS n_rows,
                   COUNT(DISTINCT athlete_uuid) AS n_athletes
            FROM public.f_proteus
            WHERE exercise_name IS NOT NULL
            GROUP BY exercise_name, movement, dominance
            ORDER BY n_rows DESC
        """)
        _print_table("f_proteus exercise_name + movement + dominance", rows,
                     cols=["exercise_name", "movement", "dominance", "n_rows", "n_athletes"])

        cols = query(conn, """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'f_proteus'
              AND data_type IN ('numeric', 'integer', 'double precision')
            ORDER BY column_name
        """)
        _print_table("f_proteus numeric columns (for picking power metric)", cols,
                     cols=["column_name", "data_type"], limit=200)


def screen() -> None:
    with backend_conn() as conn:
        for tbl in ("f_athletic_screen_cmj", "f_athletic_screen_dj",
                    "f_athletic_screen_ppu", "f_athletic_screen_slv"):
            cols = query(conn, """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                  AND column_name NOT IN
                    ('id', 'athlete_uuid', 'session_date', 'source_system',
                     'source_athlete_id', 'trial_name', 'age_at_collection',
                     'age_group', 'created_at', 'demographic', 'side')
                ORDER BY column_name
            """, [tbl])
            _print_table(f"{tbl} metric columns", cols,
                         cols=["column_name", "data_type"], limit=200)


def athletes_overview() -> None:
    """Quick sanity check on distribution of age_group across the warehouse."""
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT age_group, COUNT(*) AS n_athletes,
                   SUM(CASE WHEN has_pitching_data        THEN 1 ELSE 0 END) AS n_pitching,
                   SUM(CASE WHEN has_hitting_data         THEN 1 ELSE 0 END) AS n_hitting,
                   SUM(CASE WHEN has_athletic_screen_data THEN 1 ELSE 0 END) AS n_screen,
                   SUM(CASE WHEN has_mobility_data        THEN 1 ELSE 0 END) AS n_mobility,
                   SUM(CASE WHEN has_proteus_data         THEN 1 ELSE 0 END) AS n_proteus,
                   SUM(CASE WHEN has_pro_sup_data         THEN 1 ELSE 0 END) AS n_pro_sup
            FROM analytics.d_athletes
            GROUP BY age_group
            ORDER BY n_athletes DESC
        """)
        _print_table("d_athletes by age_group", rows,
                     cols=["age_group", "n_athletes", "n_pitching", "n_hitting",
                           "n_screen", "n_mobility", "n_proteus", "n_pro_sup"])


MODES = {
    "athletes": athletes_overview,
    "pitching": pitching,
    "hitting": hitting,
    "mobility": mobility,
    "proteus": proteus,
    "screen": screen,
}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        for a in args:
            if a not in MODES:
                print(f"Unknown mode '{a}'. Choices: {', '.join(MODES)}")
                sys.exit(2)
            MODES[a]()
    else:
        for fn in MODES.values():
            fn()
