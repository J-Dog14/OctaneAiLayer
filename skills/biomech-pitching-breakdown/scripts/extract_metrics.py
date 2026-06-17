#!/usr/bin/env python3
"""
Extract 8-cylinder metrics from a B_Young Biomechanics report.

Handles both report formats:
  1. Single combined JSON file (e.g., "Maclean,_Patrick-6a0360e1f4bcfd69dddef904.json")
  2. Split format with metadata.json + results/results.json (newer reports, often zipped)

Usage:
    python extract_metrics.py <input_path> [--output <output.json>]

  <input_path> can be:
    - a single JSON file
    - a zip file (will be unzipped and processed)
    - a directory containing metadata.json + results/results.json

  Output is a JSON file (or stdout) with structured metrics for the report builder.
"""

import argparse
import json
import os
import sys
import tempfile
import zipfile
from glob import glob
from pathlib import Path
from statistics import mean, stdev


def find_report_files(input_path):
    """Locate metadata + results JSON files from various input formats."""
    path = Path(input_path)

    # Case 1: single combined JSON file
    if path.is_file() and path.suffix == '.json':
        with open(path) as f:
            data = json.load(f)
        # Combined format has both 'measurements' and 'results' at top level
        if 'measurements' in data and 'results' in data:
            return data, data
        # Otherwise it might be just one or the other — error
        sys.exit(f"JSON file at {path} does not contain a complete report")

    # Case 2: zip file
    if path.is_file() and path.suffix == '.zip':
        tmpdir = tempfile.mkdtemp(prefix='byoung_')
        with zipfile.ZipFile(path) as z:
            z.extractall(tmpdir)
        return find_report_files(tmpdir)

    # Case 3: directory — find metadata + results
    if path.is_dir():
        meta_paths = list(path.rglob('metadata.json'))
        res_paths = list(path.rglob('results.json'))
        if not meta_paths or not res_paths:
            sys.exit(f"Could not find metadata.json + results.json under {path}")
        with open(meta_paths[0]) as f:
            metadata = json.load(f)
        with open(res_paths[0]) as f:
            results = json.load(f)
        return metadata, results

    sys.exit(f"Cannot read report from {input_path}")


def build_scalar_lookup(results_data):
    """Build a {metric_id: {measurement_name: value}} lookup from results."""
    lookup = {}
    results_list = results_data.get('results', results_data) if isinstance(results_data, dict) else results_data
    for r in results_list:
        if r.get('type') != 'scalar':
            continue
        inner = {}
        for entry in r.get('data', []):
            vals = entry.get('values')
            if vals and len(vals) > 0:
                inner[entry['measurement']] = vals[0]
        lookup[r['id']] = inner
    return lookup


def get_fastballs(metadata):
    """Return list of {id, velocity} dicts for fastballs in the report."""
    fastballs = []
    for m in metadata.get('measurements', []):
        if 'Fastball' not in str(m.get('id', '')):
            continue
        fields = {f['id']: f.get('value') for f in m.get('fields', [])}
        velocity = None
        try:
            velocity = float(str(fields.get('Comments', '')).strip())
        except (ValueError, AttributeError):
            pass
        fastballs.append({
            'id': m['id'],
            'velocity': velocity,
            'duration': m.get('duration'),
        })
    return fastballs


def avg_across(lookup, key, fb_ids):
    """Average of a metric across the listed fastball IDs. Returns None if no data."""
    vals = []
    for fid in fb_ids:
        v = lookup.get(key, {}).get(fid)
        if v is not None:
            vals.append(v)
    return mean(vals) if vals else None


def safe_div(num, denom):
    if num is None or denom is None or denom == 0:
        return None
    return num / denom


def compute_hz_abd_dwell(lookup, fb_ids, threshold=-20):
    """How long does horizontal abduction stay at or below threshold after FS?"""
    base = avg_across(lookup, 'Pitching_Shoulder_Angle@Footstrike_X', fb_ids)
    progression = {0: base}
    for ms in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120):
        progression[ms] = avg_across(lookup, f'Pitching_Shoulder_Angle@Footstrike_{ms}ms_X', fb_ids)
    # Largest ms at which value is still ≤ threshold
    dwell = 0
    for ms in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120):
        v = progression.get(ms)
        if v is not None and v <= threshold:
            dwell = ms
    return dwell, progression


def extract(metadata, results):
    """Extract all 8-cylinder + chain analysis metrics from a report."""
    lookup = build_scalar_lookup(results)
    fastballs = get_fastballs(metadata)
    fb_ids = [fb['id'] for fb in fastballs]
    velos = [fb['velocity'] for fb in fastballs if fb['velocity'] is not None]

    custom = {f['id']: f.get('value') for f in metadata.get('metadata', {}).get('customFields', [])}

    # Subject info
    subject = {
        'name': metadata['subject']['displayName'],
        'handed': 'LHP' if 'Left' in metadata['project']['subtype'] else 'RHP',
        'dob': custom.get('subject/Date of birth'),
        'height_m': float(custom.get('subject/Height') or 0) or None,
        'weight_kg': float(custom.get('subject/Weight') or 0) or None,
        'test_date': custom.get('subject/Creation date'),
        'operator': metadata.get('operator', {}).get('displayName', 'Brandon Young'),
        'lab': metadata.get('clientId', 'B_Young Biomechanics'),
    }

    velocity_summary = {
        'mean': mean(velos) if velos else None,
        'sd': stdev(velos) if len(velos) > 1 else 0.0,
        'min': min(velos) if velos else None,
        'max': max(velos) if velos else None,
        'count': len(velos),
        'throws': [{'id': fb['id'], 'velocity': fb['velocity']} for fb in fastballs],
    }

    # ===== Cylinder 1 — Stride =====
    cyl1 = {
        'stride_pct_height': avg_across(lookup, 'STRIDE_LENGTH_MEAN_PERCENT', fb_ids),
        'stride_length_m': avg_across(lookup, 'STRIDE_LENGTH', fb_ids),
    }

    # ===== Cylinder 2 — Hip-Shoulder Separation =====
    cyl2 = {
        'hss_fs': avg_across(lookup, 'Trunk Rot wrt Pelvis Rot@Footstrike', fb_ids),
        'hss_mer': avg_across(lookup, 'Trunk Rot wrt Pelvis Rot@Max_Shoulder_Rot', fb_ids),
        'hss_release': avg_across(lookup, 'Trunk Rot wrt Pelvis Rot@Release', fb_ids),
    }

    # ===== Cylinder 3 — Trunk Velocity (sagittal × transverse) =====
    sag_vel = avg_across(lookup, 'MaxTrunkLinearVel_MPH_Y', fb_ids)
    trk_flex = avg_across(lookup, 'Trunk_Angle@Release_X', fb_ids)
    cyl3 = {
        'sagittal_trunk_vel_mph': sag_vel,
        'trunk_forward_flex_rel': trk_flex,
        'product': (sag_vel * trk_flex) if (sag_vel is not None and trk_flex is not None) else None,
        # For reference — thorax angular velocity (kinematic sequence, not Cyl 3 per se)
        'thorax_ang_vel_peak': avg_across(lookup, 'Thorax_Ang_Vel_max', fb_ids),
    }

    # ===== Cylinder 4 — Front Leg Block =====
    knee_fs = avg_across(lookup, 'Lead_Knee_Angle@Footstrike_X', fb_ids)
    knee_rel = avg_across(lookup, 'Lead_Knee_Angle@Release_X', fb_ids)
    cyl4 = {
        'lead_knee_fs': knee_fs,
        'lead_knee_release': knee_rel,
        'lead_knee_extension': (knee_fs - knee_rel) if (knee_fs is not None and knee_rel is not None) else None,
        'lead_knee_mer': avg_across(lookup, 'Lead_Knee_Angle@Max_Shoulder_Rot_X', fb_ids),
        'lead_leg_vert_grf_release': avg_across(lookup, 'Lead GRF_BW@Release_Z', fb_ids),
        'lead_leg_vert_grf_fs': avg_across(lookup, 'Lead GRF_BW@Footstrike_Z', fb_ids),
    }

    # ===== Cylinder 5 — Trunk Tilt at Release =====
    cyl5 = {
        'forward_flex': avg_across(lookup, 'Trunk_Angle@Release_X', fb_ids),
        'lateral_tilt': avg_across(lookup, 'Trunk_Angle@Release_Y', fb_ids),
        'rotation_z': avg_across(lookup, 'Trunk_Angle@Release_Z', fb_ids),
    }

    # ===== Cylinder 6 — Horizontal Abduction (Arm Trail) =====
    dwell_ms, hz_progression = compute_hz_abd_dwell(lookup, fb_ids, threshold=-20)
    cyl6 = {
        'hz_abd_fs': avg_across(lookup, 'Pitching_Shoulder_Angle@Footstrike_X', fb_ids),
        'hz_abd_mer': avg_across(lookup, 'Pitching_Shoulder_Angle@Max_Shoulder_Rot_X', fb_ids),
        'hz_abd_release': avg_across(lookup, 'Pitching_Shoulder_Angle@Release_X', fb_ids),
        'dwell_ms_below_neg20': dwell_ms,
        'progression': hz_progression,
        # Arm timing — related but distinct
        'er_fs': avg_across(lookup, 'Pitching_Shoulder_Angle@Footstrike_Z', fb_ids),
        'max_er_mer': avg_across(lookup, 'Pitching_Shoulder_Angle@Max_Shoulder_Rot_Z', fb_ids),
    }

    # ===== Cylinder 7 — Drive Leg / GRF =====
    cyl7 = {
        'back_leg_grf_max': avg_across(lookup, 'Back_Leg_GRF_mag_max', fb_ids),
        'lead_leg_braking_fs': avg_across(lookup, 'Lead GRF_BW@Footstrike_Y', fb_ids),
        'lead_leg_braking_release': avg_across(lookup, 'Lead GRF_BW@Release_Y', fb_ids),
    }

    # ===== Cylinder 8 — Arm Health =====
    elbow_torque = avg_across(lookup, 'Elbow_Torque_Nm@Max_Shoulder_Rot', fb_ids)
    shoulder_dist_n = avg_across(lookup, 'Shoulder_Force_Abd_N@Max_Shoulder_Abd_Force_N', fb_ids)
    velo_mean = velocity_summary['mean']
    cyl8 = {
        'elbow_torque_mer_nm': elbow_torque,
        'elbow_torque_release_nm': avg_across(lookup, 'Elbow_Torque_Nm@Release', fb_ids),
        'max_elbow_force_bw': avg_across(lookup, 'Max_Elbow_Force', fb_ids),
        'shoulder_distraction_n': shoulder_dist_n,
        'shoulder_distraction_bw': avg_across(lookup, 'Shoulder_Force_Abd@Max_Shoulder_Abd_Force', fb_ids),
        'stress_per_mph': (elbow_torque / velo_mean) if (elbow_torque is not None and velo_mean) else None,
    }

    # ===== Kinematic sequence / chain analysis (for chain breakdown reports) =====
    pelvis_peak = avg_across(lookup, 'Pelvis_Ang_Vel_max', fb_ids)
    thorax_peak = cyl3['thorax_ang_vel_peak']
    humerus_peak = avg_across(lookup, 'Pitching_Humerus_Ang_Vel_max', fb_ids)
    hand_peak = avg_across(lookup, 'Pitching_Hand_Ang_Vel_max', fb_ids)

    fs_time = avg_across(lookup, 'FootstrikeTime', fb_ids)
    rel_time = avg_across(lookup, 'ReleaseTime', fb_ids)

    chain = {
        # Peak segment angular velocities (transverse)
        'pelvis_ang_vel_peak': pelvis_peak,
        'thorax_ang_vel_peak': thorax_peak,
        'humerus_ang_vel_peak': humerus_peak,
        'shoulder_ang_vel_peak': avg_across(lookup, 'Pitching_Shoulder_Ang_Vel_max', fb_ids),
        'elbow_ang_vel_peak': avg_across(lookup, 'Pitching_Elbow_Ang_Vel_max', fb_ids),
        'hand_ang_vel_peak': hand_peak,
        # Amplification ratios
        'amp_pelvis_to_thorax': safe_div(thorax_peak, pelvis_peak),
        'amp_thorax_to_humerus': safe_div(humerus_peak, thorax_peak),
        'amp_humerus_to_hand': safe_div(hand_peak, humerus_peak),
        # Linear velocities (mph)
        'pelvis_lin_x_mph': avg_across(lookup, 'MaxPelvisLinearVel_MPH_X', fb_ids),
        'pelvis_lin_y_mph': avg_across(lookup, 'MaxPelvisLinearVel_MPH_Y', fb_ids),
        'pelvis_lin_z_mph': avg_across(lookup, 'MaxPelvisLinearVel_MPH_Z', fb_ids),
        'trunk_lin_x_mph': avg_across(lookup, 'MaxTrunkLinearVel_MPH_X', fb_ids),
        'trunk_lin_y_mph': sag_vel,
        'trunk_lin_z_mph': avg_across(lookup, 'MaxTrunkLinearVel_MPH_Z', fb_ids),
        # Peak timing relative to FS
        'pelvis_peak_time': avg_across(lookup, 'MaxPelvisVelTime', fb_ids),
        'thorax_peak_time': avg_across(lookup, 'MaxThoraxVelTime', fb_ids),
        'humerus_peak_time': avg_across(lookup, 'MaxHumerusVelTime', fb_ids),
        'hand_peak_time': avg_across(lookup, 'MaxHandVelTime', fb_ids),
        'fs_time': fs_time,
        'release_time': rel_time,
    }
    # Derived: peak time relative to FS
    if fs_time is not None:
        chain['pelvis_peak_rel_fs_ms'] = (chain['pelvis_peak_time'] - fs_time) * 1000 if chain['pelvis_peak_time'] else None
        chain['thorax_peak_rel_fs_ms'] = (chain['thorax_peak_time'] - fs_time) * 1000 if chain['thorax_peak_time'] else None
        chain['humerus_peak_rel_fs_ms'] = (chain['humerus_peak_time'] - fs_time) * 1000 if chain['humerus_peak_time'] else None
        chain['hand_peak_rel_fs_ms'] = (chain['hand_peak_time'] - fs_time) * 1000 if chain['hand_peak_time'] else None

    # ===== Pelvis decel mechanism =====
    pelv_av_fs = avg_across(lookup, 'Pelvis_Ang_Vel@Footstrike_X', fb_ids)
    pelv_av_mer = avg_across(lookup, 'Pelvis_Ang_Vel@Max_Shoulder_Rot_X', fb_ids)
    pelv_stop_time = avg_across(lookup, 'PelvisRot_StopTime', fb_ids)
    bk_fs = avg_across(lookup, 'Back_Knee_Angle@Footstrike_X', fb_ids)
    bk_mer = avg_across(lookup, 'Back_Knee_Angle@Max_Shoulder_Rot_X', fb_ids)
    decel = {
        'pelv_ang_vel_mkh': avg_across(lookup, 'Pelvis_Ang_Vel@MaxKneeHeight_X', fb_ids),
        'pelv_ang_vel_fs': pelv_av_fs,
        'pelv_ang_vel_mer': pelv_av_mer,
        'pelv_stop_after_fs_ms': (pelv_stop_time - fs_time) * 1000 if (pelv_stop_time and fs_time) else None,
        'back_knee_mkh': avg_across(lookup, 'Back_Knee_Angle@MaxKneeHeight_X', fb_ids),
        'back_knee_fs': bk_fs,
        'back_knee_mer': bk_mer,
        'back_knee_release': avg_across(lookup, 'Back_Knee_Angle@Release_X', fb_ids),
        'delta_back_knee_fs_to_mer': (bk_mer - bk_fs) if (bk_fs is not None and bk_mer is not None) else None,
    }

    return {
        'subject': subject,
        'velocity': velocity_summary,
        'cyl1': cyl1,
        'cyl2': cyl2,
        'cyl3': cyl3,
        'cyl4': cyl4,
        'cyl5': cyl5,
        'cyl6': cyl6,
        'cyl7': cyl7,
        'cyl8': cyl8,
        'chain': chain,
        'pelvis_decel': decel,
        'fastball_ids': fb_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='B_Young report — JSON, zip, or directory')
    parser.add_argument('--output', '-o', default=None, help='Output JSON path (default: stdout)')
    args = parser.parse_args()

    metadata, results = find_report_files(args.input_path)
    extracted = extract(metadata, results)

    output_json = json.dumps(extracted, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == '__main__':
    main()