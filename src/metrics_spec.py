"""
Locked-down metric spec for the athlete deficit profile.

Each METRICS entry defines:
- key:        canonical key used in raw_values / z_scores JSONB
- modality:   high-level bucket; controls which session_date the value comes from
- applies_to: 'pitcher', 'hitter', or 'both'
- extract:    extraction recipe (see profiler.py for the dispatcher)

Adding or renaming a metric here is the only thing required to add it to every
athlete's profile. Norms get recomputed by src.refresh_norms.

NOTE on naming: "sittiing_t_spine_pvc_*" reflects a real typo in the DB
column name — DO NOT correct it here. The string must match the column exactly.

Extract spec options:
  pt_json / ht_json:
      key:   JSON path (with axis suffix already, e.g. "PROCESSED.X.Z")
             OR a parent path + axis kwarg (we concatenate with a dot).
      axis:  optional. If set, the lookup key is f"{key}.{axis}".
      scale: optional float. The extracted value is multiplied by this (e.g. 1000
             to convert seconds to ms).
      abs:   optional bool. If true, the absolute value is taken.

  pt_json_diff / ht_json_diff:
      minuend_key, subtrahend_key: both keys
      axis: optional, applied to both
      scale: optional, applied to (minuend - subtrahend)

  mob_col:  column on f_mobility
  proteus_movement: as before
  screen_col / screen_col_side: as before
"""
from __future__ import annotations

# ─────────────────────────── Shared constants ────────────────────────────────

_PROTEUS_VARS = [
    "power_high", "power_mean",
    "velocity_high", "velocity_mean",
    "acceleration_high", "acceleration_mean",
]

# Screen columns to pull per movement (CMJ / DJ / PPU / SLV).
# DJ adds rsi and ct (contact time) on top of the shared set.
_SCREEN_COLS_BASE = [
    "jh_in", "pp_w_per_kg", "vel_at_pp", "force_at_pp",
    "auc_j", "kurtosis", "rpd_max_w_per_s", "time_to_rpd_max_s",
]
_SCREEN_COLS_DJ_EXTRA = ["rsi", "ct"]

# ─────────────────────── Mobility column rules ──────────────────────────────
# Each column has a "rule" dict describing how to handle mixed legacy 1-3
# grades and modern goniometer / dynamometer measurements:
#
#   midpoints: maps integer 1/2/3 grade -> equivalent ROM (degrees) or MMT (%BW)
#              value. Pulled from Mobility Thresholds doc.
#              The CASE in the SQL uses these for legacy entries (where the
#              stored value is exactly 1, 2, or 3) and passes the raw value
#              through unchanged for modern measurements (>3 or non-integer).
#   no midpoints = pure pass-through (already on a single consistent scale,
#                  OR pure 1-3 scale we keep as-is).
#
# DELETED columns from the thresholds doc (shoulder_total_arc, supine_shoulder_flexion)
# are intentionally excluded.

_MOBILITY_RULES: dict[str, dict] = {
    # ── Cervical ROM (degrees) ──
    "cervical_rotation":                 {"midpoints": {1: 55, 2: 65, 3: 75}},
    "cervical_flexion":                  {"midpoints": {1: 37, 2: 42, 3: 48}},
    "cervical_extension":                {"midpoints": {1: 55, 2: 65, 3: 75}},
    "cervical_lateral_flexion":          {"midpoints": {1: 37, 2: 42, 3: 48}},
    # ── Shoulder ROM (degrees) ──
    "horizontal_abduction":              {"midpoints": {1: 27, 2: 32, 3: 38}},
    "shoulder_ir":                       {"midpoints": {1: 27, 2: 35, 3: 42}},
    "shoulder_er":                       {"midpoints": {1: 112, 2: 120, 3: 128}},
    # ── Shoulder stability MMT (% BW) — DB col has the "abuduction" typo ──
    "shoulder_stability_flexion":                       {"midpoints": {1: 11, 2: 17.5, 3: 25}},
    "shoulder_stability_abduction":                     {"midpoints": {1: 10, 2: 15, 3: 21}},
    "shoulder_stability_er_at_0_deg_horiz_abuduction":  {"midpoints": {1: 6,  2: 10, 3: 14}},  # typo
    "shoulder_stability_ir_at_0_deg_horiz_abduction":   {"midpoints": {1: 8,  2: 13, 3: 18}},
    # ── Elbow / forearm ROM (degrees) ──
    # Elbow extension: 0° = full extension (best), positive = loss-of-extension (worse).
    # So 1=worst (~7° lag), 2=middle (~3° lag), 3=full (0°). This is inverted from others.
    "elbow_extension":                   {"midpoints": {1: 7,   2: 3,   3: 0}},
    "elbow_flexion":                     {"midpoints": {1: 122, 2: 130, 3: 138}},
    "elbow_pronation":                   {"midpoints": {1: 65,  2: 71,  3: 78}},
    "elbow_supination":                  {"midpoints": {1: 67,  2: 75,  3: 83}},
    # ── Grip / hand MMT (% BW) ──
    "grip_strength_r":                   {"midpoints": {1: 45, 2: 57, 3: 70}},
    "gs_l":                              {"midpoints": {1: 45, 2: 57, 3: 70}},
    "grip_strength_r_at_90":             {"midpoints": {1: 40, 2: 52, 3: 63}},
    "gs_l_at_90":                        {"midpoints": {1: 40, 2: 52, 3: 63}},
    # ── Ankle (deg + MMT) ──
    "ankle_dorsiflextion_to_wall":       {"midpoints": {1: 27, 2: 32, 3: 38}},  # typo intentional
    "ankle_manual_test":                 {"midpoints": {1: 27, 2: 32, 3: 38}},  # MMT %BW
    # ── T-spine ROM (degrees) ──
    "sittiing_t_spine_pvc_r":            {"midpoints": {1: 57, 2: 67, 3: 78}},  # typo intentional
    "sittiing_t_spine_pvc_l":            {"midpoints": {1: 57, 2: 68, 3: 78}},
    # ── Hip ROM (degrees) ──
    "hamstring_stretch":                 {"midpoints": {1: 67, 2: 75, 3: 83}},
    "r_prone_hip_ir":                    {"midpoints": {1: 27, 2: 32, 3: 38}},
    "r_prone_hip_er":                    {"midpoints": {1: 27, 2: 33, 3: 38}},
    "l_prone_hip_ir":                    {"midpoints": {1: 27, 2: 33, 3: 38}},
    "l_prone_hip_er":                    {"midpoints": {1: 27, 2: 33, 3: 38}},
    # ── Lower body MMT (% BW) ──
    "prone_hamstring_raise":             {"midpoints": {1: 17, 2: 22, 3: 28}},
    "glute_strength_test_prone_hammy_push": {"midpoints": {1: 27, 2: 32, 3: 38}},
    "mid_trap":                          {"midpoints": {1: 7, 2: 12, 3: 17}},
    "low_trap":                          {"midpoints": {1: 7, 2: 12, 3: 17}},

    # ── Pure 1-3 scale: subjective tests, no degree/%BW conversion ──
    "back_to_wall_shoulder_flexion":     {},
    "pelvic_tilt_against_wall":          {},
    "radial_nerve_glide":                {},
    "ulnar_nerve_glide":                 {},
    "backbend":                          {},
    "slump_test":                        {},
    "thomas_test_hip_flexor_r":          {},
    "thomas_test_hip_flexor_l":          {},
    "young_stretch_passive":             {},
    "hip_pinch":                         {},

    # ── Other (raw degrees, no scale conversion) ──
    "isa":                               {},
}


# ─────────────────────────── METRICS list ────────────────────────────────────

METRICS: list[dict] = []


# ══════════════════════════ PITCHING ════════════════════════════════════════
# f_pitching_trials.metrics  —  flat dotted JSON keys, values may be [n] or n

_pitch = [
    # ── shoulder kinematics ──
    ("pitch_max_horizontal_abduction",      "PROCESSED.Pitching_Shoulder_Angle@Footstrike", "X"),
    ("pitch_arm_er_timing_at_fc_xyz",       "PROCESSED.Pitching_Shoulder_Angle_XYZ@Footstrike", "Z"),
    ("pitch_arm_timing_at_fc_simple",       "PROCESSED.Pitching_Shoulder_Angle@Footstrike", "Z"),
    ("pitch_max_external_rotation_xyz",     "PROCESSED.Pitching_Shoulder_Angle_XYZ@Max_Shoulder_Rot", "Z"),
    ("pitch_max_external_rotation_simple",  "PROCESSED.Pitching_Shoulder_Angle_Max", "Z"),
    ("pitch_peak_abduction",                "PROCESSED.Pitching_Shoulder_Angle_Min", "X"),

    # ── pelvis / trunk at events ──
    ("pitch_pelvis_rotation_at_fc",  "PROCESSED.Pelvis_Angle@Footstrike", "Z"),
    ("pitch_trunk_rotation_at_fc",   "PROCESSED.Trunk_Angle@Footstrike",  "Z"),
    ("pitch_trunk_flexion_at_release", "PROCESSED.Trunk_Angle@Release",   "X"),

    # ── lead leg / knee ──
    ("pitch_lead_knee_at_fc",       "PROCESSED.Lead_Knee_Angle@Footstrike", "X"),
    ("pitch_lead_knee_at_release",  "PROCESSED.Lead_Knee_Angle@Release",    "X"),

    # ── hip-shoulder separation at FC ──
    ("pitch_hip_shoulder_sep_at_fc", "PROCESSED.Hip Shoulders Sep@Footstrike", "Z"),  # space in key

    # (ball release speed is pulled from a column, not JSON — added separately below)

    # ── kinematic sequence (max angular velocities) ──
    ("pitch_pelvis_ang_vel_max",   "KINEMATIC_SEQUENCE.Pelvis_Ang_Vel_max",            "X"),
    ("pitch_thorax_ang_vel_max",   "KINEMATIC_SEQUENCE.Thorax_Ang_Vel_max",            "X"),
    ("pitch_humerus_ang_vel_max",  "KINEMATIC_SEQUENCE.Pitching_Humerus_Ang_Vel_max",  "X"),
    ("pitch_hand_ang_vel_max",     "KINEMATIC_SEQUENCE.Pitching_Hand_Ang_Vel_max",     "X"),

    # ── timing of max segment velocities ──
    ("pitch_time_max_pelvis_vel",  "TIMING.MaxPelvisVelTime",  "X"),
    ("pitch_time_max_thorax_vel",  "TIMING.MaxThoraxVelTime",  "X"),
    ("pitch_time_max_humerus_vel", "TIMING.MaxHumerusVelTime", "X"),
    ("pitch_time_max_hand_vel",    "TIMING.MaxHandVelTime",    "X"),

    # ── lead leg GRF ──
    ("pitch_lead_leg_grf_mag_midpoint", "PROCESSED.Lead_Leg_GRF_mag_Midpoint_FS_Release", "X"),
    ("pitch_lead_leg_grf_mag_max",      "PROCESSED.Lead_Leg_GRF_mag_max",                 "X"),
    ("pitch_lead_leg_grf_max_z",        "PROCESSED.Lead_Leg_GRF_max",                     "Z"),

    # ── center of mass pelvis linear velocity ──
    ("pitch_max_pelvis_linear_vel_mph", "PROCESSED.MaxPelvisLinearVel_MPH", "X"),

    # ── Cylinder 1: stride ──
    ("pitch_stride_length",                   "STRIDE_LENGTH.STRIDE_LENGTH", "X"),
    ("pitch_stride_length_mean_percent",      "STRIDE_LENGTH.STRIDE_LENGTH_MEAN_PERCENT", "X"),

    # ── Cylinder 2: trunk-pelvis separation scalar at FC ──
    ("pitch_trunk_rot_wrt_pelvis_at_fc",      "PROCESSED.Trunk Rot wrt Pelvis Rot@Footstrike", "X"),

    # ── Cylinder 3: trunk linear velocity (Y axis = forward direction) ──
    ("pitch_max_trunk_linear_vel_mph_y",      "PROCESSED.MaxTrunkLinearVel_MPH", "Y"),
    ("pitch_trunk_cog_vel_at_release_mph_y",  "PROCESSED.Trunk_COG_Vel@Release_MPH", "Y"),

    # ── Cylinder 5: trunk lateral tilt (Y axis) ──
    ("pitch_trunk_lateral_at_release",        "PROCESSED.Trunk_Angle@Release",   "Y"),
    ("pitch_trunk_lateral_at_fc",             "PROCESSED.Trunk_Angle@Footstrike", "Y"),

    # ── Cylinder 7: back leg GRF magnitude max ──
    ("pitch_back_leg_grf_mag_max",            "PROCESSED.Back_Leg_GRF_mag_max", "X"),

    # ── Cylinder 8: arm health (torques + shoulder force) ──
    ("pitch_elbow_torque_nm_at_max_er",       "PROCESSED.Elbow_Torque_Nm@Max_Shoulder_Rot", "X"),
    ("pitch_max_elbow_varus_torque_nm",       "PROCESSED.Max_Elbow_Varus_Torque_Nm", "X"),
    ("pitch_shoulder_force_abd_at_max",       "PROCESSED.Shoulder_Force_Abd@Max_Shoulder_Abd_Force", "X"),

    # ── Pelvis deceleration mechanism ──
    ("pitch_pelvis_ang_vel_at_fc",            "PROCESSED.Pelvis_Ang_Vel@Footstrike",            "X"),
    ("pitch_pelvis_ang_vel_at_max_er",        "PROCESSED.Pelvis_Ang_Vel@Max_Shoulder_Rot",      "X"),
    ("pitch_pelvis_rot_stop_time",            "TIMING.PelvisRot_StopTime",                      "X"),
    ("pitch_back_knee_at_fc",                 "PROCESSED.Back_Knee_Angle@Footstrike",           "X"),
    ("pitch_back_knee_at_max_er",             "PROCESSED.Back_Knee_Angle@Max_Shoulder_Rot",     "X"),
    ("pitch_back_knee_at_release",            "PROCESSED.Back_Knee_Angle@Release",              "X"),

    # ── Pelvic obliquity raw fields (Y axis of pelvis angle at events) ──
    ("pitch_pelvis_obliquity_at_fc",          "PROCESSED.Pelvis_Angle@Footstrike", "Y"),
    ("pitch_pelvis_obliquity_at_release",     "PROCESSED.Pelvis_Angle@Release",    "Y"),

    # ── Timing completeness ──
    ("pitch_release_time",                    "TIMING.ReleaseTime",                            "X"),
    ("pitch_max_shoulder_rot_time",           "TIMING.Max_Shoulder_RotTime",                   "X"),
    ("pitch_time_max_pelvis_to_max_thorax",   "PROCESSED.time_MaxPelvisVel_MaxThoraxVel",      "X"),
    ("pitch_max_elbow_vel_time",              "TIMING.MaxElbowVelTime",                        "X"),

    # ── Shoulder context (Y axis variants + elbow at max ER + elbow vel) ──
    ("pitch_shoulder_y_at_fc",                "PROCESSED.Pitching_Shoulder_Angle@Footstrike",       "Y"),
    ("pitch_shoulder_y_at_max_er",            "PROCESSED.Pitching_Shoulder_Angle@Max_Shoulder_Rot", "Y"),
    ("pitch_elbow_flex_at_max_er",            "PROCESSED.Pitching_Elbow_Angle@Max_Shoulder_Rot",    "X"),
    ("pitch_elbow_ang_vel_max",               "KINEMATIC_SEQUENCE.Pitching_Elbow_Ang_Vel_max",      "X"),

    # ── Progression extensions: the 120ms time point we want to track ──
    ("pitch_shoulder_at_fc_120ms_x",          "INCREMENT.Pitching_Shoulder_Angle@Footstrike_120ms", "X"),
    ("pitch_hss_at_fc_120ms_z",               "INCREMENT.Hip Shoulders Sep@Footstrike_120ms",       "Z"),
]

for _key, _json_key, _axis in _pitch:
    METRICS.append({
        "key": _key, "modality": "pitching", "applies_to": "pitcher",
        "extract": {"type": "pt_json", "key": _json_key,
                    **({"axis": _axis} if _axis else {})},
    })

# GRF min Y axis as absolute value (per "abs(value) taken" note)
METRICS.append({
    "key": "pitch_lead_leg_grf_min_y_abs",
    "modality": "pitching", "applies_to": "pitcher",
    "extract": {"type": "pt_json",
                "key": "PROCESSED.Lead_Leg_GRF_min",
                "axis": "Y", "abs": True},
})

# Ball release speed lives on the f_pitching_trials.velocity_mph COLUMN,
# not in the metrics JSON.
METRICS.append({
    "key": "pitch_ball_release_speed",
    "modality": "pitching", "applies_to": "pitcher",
    "extract": {"type": "pt_col", "column": "velocity_mph"},
})

# ── Pitching force plate metrics (separate table f_pitching_force_metrics) ──
# These get their own modality so we look up the right session date independently
# from biomech (in case force collection and 3D capture happen on different days).
_pitch_force_table = "f_pitching_force_metrics"
_pitch_force = [
    ("pitch_force_lead_peak_vertical_bw",        "lead_peak_vertical_bw"),
    ("pitch_force_lead_peak_braking_bw",         "lead_peak_braking_bw"),
    ("pitch_force_lead_peak_resultant_bw",       "lead_peak_resultant_bw"),
    ("pitch_force_peak_v_to_b_lag_ms",           "peak_v_to_peak_b_lag_ms"),
    ("pitch_force_lead_peak_vertical_pct_fc_br", "lead_peak_vertical_pct_fc_br"),
    ("pitch_force_lead_time_to_peak_fz_ms",      "lead_time_to_peak_fz_ms"),
    ("pitch_force_lead_rfd_vertical_bw_per_s",   "lead_rfd_vertical_bw_per_s"),
    ("pitch_force_lead_rfd_braking_bw_per_s",    "lead_rfd_braking_bw_per_s"),
    ("pitch_force_lead_impulse_v_into_ball_bws", "lead_impulse_v_into_ball_bws"),
    ("pitch_force_lead_impulse_b_into_ball_bws", "lead_impulse_b_into_ball_bws"),
    ("pitch_force_lead_fz_at_midpoint_bw",       "lead_fz_at_midpoint_bw"),
]
for _key, _col in _pitch_force:
    METRICS.append({
        "key": _key, "modality": "pitching_force", "applies_to": "pitcher",
        "extract": {"type": "pt_col", "table": _pitch_force_table, "column": _col},
    })

# Boolean flag — cast to int so AVG yields the proportion of trials with peaks-before-MER
METRICS.append({
    "key": "pitch_force_peaks_before_mer", "modality": "pitching_force",
    "applies_to": "pitcher",
    "extract": {"type": "pt_col", "table": _pitch_force_table,
                "column": "peaks_before_mer_flag", "bool_to_int": True},
})

# ── Pitching computed diffs ──
METRICS += [
    {"key": "pitch_front_leg_extension", "modality": "pitching", "applies_to": "pitcher",
     "extract": {"type": "pt_json_diff",
                 "minuend_key": "PROCESSED.Lead_Knee_Angle@Footstrike",
                 "subtrahend_key": "PROCESSED.Lead_Knee_Angle@Release", "axis": "X"}},
    {"key": "pitch_pelvic_obliquity_total", "modality": "pitching", "applies_to": "pitcher",
     "extract": {"type": "pt_json_diff",
                 "minuend_key": "PROCESSED.Pelvis_Angle@Release",
                 "subtrahend_key": "PROCESSED.Pelvis_Angle@Footstrike", "axis": "Y"}},
    {"key": "pitch_total_trunk_flexion", "modality": "pitching", "applies_to": "pitcher",
     "extract": {"type": "pt_json_diff",
                 "minuend_key": "PROCESSED.Trunk_Angle@Release",
                 "subtrahend_key": "PROCESSED.Trunk_Angle@Footstrike", "axis": "X"}},
    # Time-to-max-abduction in ms (timing fields are in seconds, hence ×1000)
    {"key": "pitch_time_to_max_abduction_ms", "modality": "pitching", "applies_to": "pitcher",
     "extract": {"type": "pt_json_diff",
                 "minuend_key": "TIMING.MaxShoulderHorAngleTime",
                 "subtrahend_key": "TIMING.FootstrikeTime", "axis": "X",
                 "scale": 1000.0}},
]


# ══════════════════════════ HITTING ═════════════════════════════════════════
# f_hitting_trials.metrics  —  flat scalar keys (no .X/.Y/.Z usually).
# Helper: many metrics come in both raw and "_MEAN" form — add both.

def _hit_with_mean(base_key: str, *, json_key: str, axis: str | None = None) -> list[dict]:
    rows = [{
        "key": base_key, "modality": "hitting", "applies_to": "hitter",
        "extract": {"type": "ht_json", "key": json_key,
                    **({"axis": axis} if axis else {})},
    }]
    rows.append({
        "key": f"{base_key}_mean", "modality": "hitting", "applies_to": "hitter",
        "extract": {"type": "ht_json", "key": f"{json_key}_MEAN",
                    **({"axis": axis} if axis else {})},
    })
    return rows

# Single-shot (no _MEAN counterpart)
_hit_single = [
    ("hit_horizontal_attack_angle_at_contact", "PLANE.Horizontal_attack_angle"),
    ("hit_vertical_attack_angle_at_contact",   "PLANE.Vertical_attack_angle"),
    ("hit_bat_angle_frontal_at_contact",       "PLANE.Bat_Angle_Frontal@Contact"),
    ("hit_bat_angle_sagittal_at_contact",      "PLANE.Bat_Angle_Sagittal@Contact"),
    ("hit_bat_angle_transversal_at_contact",   "PLANE.Bat_Angle_Transversal@Contact"),
    ("hit_bat_travelled_distance_max",         "PROCESSED.Bat_travelled_distance_max"),
    ("hit_pelvis_at_contact",                  "PROCESSED.Pelvis_Angle@Contact"),
    ("hit_trunk_at_contact",                   "PROCESSED.Trunk_Angle@Contact"),
    ("hit_hip_shoulder_sep_at_setup",          "PROCESSED.Pelvis_Shoulders_Separation@Setup"),
]
for _key, _json_key in _hit_single:
    METRICS.append({
        "key": _key, "modality": "hitting", "applies_to": "hitter",
        "extract": {"type": "ht_json", "key": _json_key},
    })

# Metrics that have both base + _MEAN forms
_hit_pairs = [
    ("hit_max_pelvis_ang_vel",                "PROCESSED.Max_Pelvis_Ang_Vel"),
    ("hit_max_thorax_ang_vel",                "PROCESSED.Max_Thorax_Ang_Vel"),
    ("hit_max_lead_forearm_ang_vel",          "PROCESSED.Max_Lead_Forearm_Ang_Vel"),
    ("hit_max_lead_hand_ang_vel",             "PROCESSED.Max_Lead_Hand_Ang_Vel"),
    ("hit_max_bat_ang_vel",                   "PROCESSED.Max_Bat_Ang_Vel"),
    ("hit_max_rpv_cgpos_linear_vel",          "PROCESSED.Max_RPV_CGPos_VLab_Linear_Vel"),
    ("hit_max_rta_cgpos_linear_vel",          "PROCESSED.Max_RTA_CGPos_VLab_Linear_Vel"),
    ("hit_lead_knee_at_lead_foot_down",       "PROCESSED.Lead_Knee_Angle@Lead_Foot_Down"),
    ("hit_lead_knee_at_contact",              "PROCESSED.Lead_Knee_Angle@Contact"),
    ("hit_lead_knee_ang_vel_ext_max",         "PROCESSED.Lead_Knee_Ang_Vel_ext_max"),
    ("hit_pelvis_at_lead_foot_down",          "PROCESSED.Pelvis_Angle@Lead_Foot_Down"),
    ("hit_pelvis_shoulders_sep_at_lead_foot_down",     "PROCESSED.Pelvis_Shoulders_Separation@Lead_Foot_Down"),
    ("hit_pelvis_shoulders_sep_at_downswing",          "PROCESSED.Pelvis_Shoulders_Separation@Downswing"),
    ("hit_pelvis_shoulders_sep_at_max_bat_ang_vel",    "PROCESSED.Pelvis_Shoulders_Separation@Max_Bat_Ang_Vel"),
    ("hit_pelvis_shoulders_sep_at_max_lead_hand_ang_vel", "PROCESSED.Pelvis_Shoulders_Separation@Max_Lead_Hand_Ang_Vel"),
    ("hit_pelvis_shoulders_sep_at_contact",            "PROCESSED.Pelvis_Shoulders_Separation@Contact"),
    ("hit_trunk_at_lead_foot_down",           "PROCESSED.Trunk_Angle@Lead_Foot_Down"),
    ("hit_stride_width_at_lead_foot_down",    "PROCESSED.Stride_Width@Lead_Foot_Down"),
]
for _key, _json_key in _hit_pairs:
    METRICS += _hit_with_mean(_key, json_key=_json_key)

# ── Hitting computed diffs ──
METRICS += [
    # Lead leg block (knee extension during the swing) — already canonical name
    {"key": "hit_lead_leg_block_delta", "modality": "hitting", "applies_to": "hitter",
     "extract": {"type": "ht_json_diff",
                 "minuend_key":    "PROCESSED.Lead_Knee_Angle@Contact",
                 "subtrahend_key": "PROCESSED.Lead_Knee_Angle@Lead_Foot_Down"}},
    {"key": "hit_pelvis_total_rotation", "modality": "hitting", "applies_to": "hitter",
     "extract": {"type": "ht_json_diff",
                 "minuend_key":    "PROCESSED.Pelvis_Angle@Contact",
                 "subtrahend_key": "PROCESSED.Pelvis_Angle@Lead_Foot_Down"}},
    {"key": "hit_trunk_total_rotation", "modality": "hitting", "applies_to": "hitter",
     "extract": {"type": "ht_json_diff",
                 "minuend_key":    "PROCESSED.Trunk_Angle@Contact",
                 "subtrahend_key": "PROCESSED.Trunk_Angle@Lead_Foot_Down"}},
]


# ══════════════════════════ MOBILITY ════════════════════════════════════════
for _col, _rule in _MOBILITY_RULES.items():
    _extract = {"type": "mob_col", "column": _col}
    if _rule.get("midpoints"):
        _extract["midpoints"] = _rule["midpoints"]
    METRICS.append({
        "key": f"mob_{_col}",
        "modality": "mobility", "applies_to": "both",
        "extract": _extract,
    })


# ══════════════════════════ PROTEUS ═════════════════════════════════════════
# Proteus pitcher: Shot Put + D2 Extension × 6 variables
for _move_key, _move_filter in [("shotput",      {"movement_like": "Shot Put"}),
                                 ("d2_extension", {"movement_like": "D2 Extension"})]:
    for _var in _PROTEUS_VARS:
        METRICS.append({
            "key":        f"proteus_pitcher_{_move_key}_{_var}",
            "modality":   "proteus_pitcher", "applies_to": "pitcher",
            "extract": {
                "type": "proteus_movement", "position_like": "pitcher",
                "value_col": _var, **_move_filter,
            },
        })

# Proteus hitter: Shot Put + Straight Arm Trunk Rotation × 6 variables
for _move_key, _move_filter in [("shotput",        {"movement_like": "Shot Put"}),
                                 ("trunk_rotation", {"movement_eq":  "Straight Arm Trunk Rotation"})]:
    for _var in _PROTEUS_VARS:
        METRICS.append({
            "key":        f"proteus_hitter_{_move_key}_{_var}",
            "modality":   "proteus_hitter", "applies_to": "hitter",
            "extract": {
                "type": "proteus_movement", "position_not_like": "pitcher",
                "value_col": _var, **_move_filter,
            },
        })


# ══════════════════════════ ATHLETIC SCREEN ═════════════════════════════════
# CMJ / PPU share the base column set; DJ adds rsi + ct; SLV uses side filter.
for _m in ["cmj", "ppu"]:
    for _c in _SCREEN_COLS_BASE:
        METRICS.append({
            "key":        f"screen_{_m}_{_c}",
            "modality":   f"athletic_screen_{_m}", "applies_to": "both",
            "extract": {"type": "screen_col",
                        "table": f"f_athletic_screen_{_m}", "column": _c},
        })

for _c in _SCREEN_COLS_BASE + _SCREEN_COLS_DJ_EXTRA:
    METRICS.append({
        "key":        f"screen_dj_{_c}",
        "modality":   "athletic_screen_dj", "applies_to": "both",
        "extract": {"type": "screen_col",
                    "table": "f_athletic_screen_dj", "column": _c},
    })

# SLV left vs right
for _side in ["left", "right"]:
    for _c in _SCREEN_COLS_BASE:
        METRICS.append({
            "key":        f"screen_slv_{_side}_{_c}",
            "modality":   "athletic_screen_slv", "applies_to": "both",
            "extract": {"type": "screen_col_side",
                        "table": "f_athletic_screen_slv",
                        "column": _c, "side": _side},
        })


# ─── Maps each modality to a "primary" SQL recipe for latest-session-date ──
MODALITY_SESSION_TABLE: dict[str, dict] = {
    "pitching":            {"sql": "SELECT MAX(session_date) FROM public.f_pitching_trials "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "pitching_force":      {"sql": "SELECT MAX(session_date) FROM public.f_pitching_force_metrics "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "hitting":             {"sql": "SELECT MAX(session_date) FROM public.f_hitting_trials "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "mobility":            {"sql": "SELECT MAX(session_date) FROM public.f_mobility "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "proteus_pitcher":     {"sql": "SELECT MAX(session_date) FROM public.f_proteus "
                                   "WHERE athlete_uuid = %s AND session_date <= %s "
                                   "AND position ILIKE 'pitcher%%'"},
    "proteus_hitter":      {"sql": "SELECT MAX(session_date) FROM public.f_proteus "
                                   "WHERE athlete_uuid = %s AND session_date <= %s "
                                   "AND (position IS NULL OR position NOT ILIKE 'pitcher%%')"},
    "athletic_screen_cmj": {"sql": "SELECT MAX(session_date) FROM public.f_athletic_screen_cmj "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "athletic_screen_dj":  {"sql": "SELECT MAX(session_date) FROM public.f_athletic_screen_dj "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "athletic_screen_ppu": {"sql": "SELECT MAX(session_date) FROM public.f_athletic_screen_ppu "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
    "athletic_screen_slv": {"sql": "SELECT MAX(session_date) FROM public.f_athletic_screen_slv "
                                   "WHERE athlete_uuid = %s AND session_date <= %s"},
}


def get_metrics_for_role(role: str) -> list[dict]:
    """role in {'pitcher', 'hitter', 'both'}. 'both' returns everything."""
    if role == "both":
        return list(METRICS)
    return [m for m in METRICS if m["applies_to"] in (role, "both")]


def metric_keys() -> list[str]:
    return [m["key"] for m in METRICS]
