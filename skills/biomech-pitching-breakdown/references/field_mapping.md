# Visual3D / B_Young Field Mappings

The B_Young Biomechanics reports are processed in Qualisys/Visual3D 4.6. Metric names follow this pattern:

```
<Joint/Segment>_<MotionType>@<Event>_<Axis>
```

For example: `Lead_Knee_Angle@Footstrike_X` = lead knee angle at footstrike, X axis.

## Axis conventions

- **X axis:** Sagittal plane motion (flexion/extension). For joints with rotation, X is often horizontal abduction for the shoulder. Always verify against expected magnitudes.
- **Y axis:** Frontal plane motion (abduction/adduction, lateral tilt). For pelvis at MKH/FS, this is lateral tilt.
- **Z axis:** Transverse plane motion (internal/external rotation, body rotation). For pelvis, this is body rotation about vertical axis.

## Event labels in the report

| Event | What it marks |
|---|---|
| `Setup` | Stance position before motion |
| `MaxKneeHeight` | Top of leg lift |
| `Footstrike` | Lead foot contacts ground (most-anchored event) |
| `Max_Shoulder_Rot` | Peak external rotation of throwing shoulder |
| `PelvisRot_Stop` | Pelvis stops rotating |
| `Release` | Ball leaves the hand |
| `Release100msAfter` | Follow-through reference |
| `Max_Shoulder_Int_Rot_FollowThru` | Peak internal rotation after release |

Many metrics also support `@Footstrike_10ms_*` through `@Footstrike_120ms_*` for time-series tracking after footstrike. Use this for horizontal abduction dwell time.

## Key fields per cylinder

### Cylinder 1 — Stride
- `STRIDE_LENGTH` (raw, in meters)
- `STRIDE_LENGTH_MEAN_PERCENT` (% of standing height)

### Cylinder 2 — Hip-Shoulder Separation
- `Trunk Rot wrt Pelvis Rot@Footstrike` (total separation, deg)
- `Trunk Rot wrt Pelvis Rot@Max_Shoulder_Rot`
- `Trunk Rot wrt Pelvis Rot@Release`
- `Hip Shoulders Sep@Footstrike_X/Y/Z` (decomposed per axis)
- `Hip Shoulders Sep@Footstrike_<10..120>ms_*` (post-FS progression)

### Cylinder 3 — Trunk Velocity (sagittal × transverse)
**Sagittal (forward) trunk velocity:**
- `MaxTrunkLinearVel_MPH_Y` (Y axis = forward toward home)
- `Trunk_COG_Vel@Release_MPH_Y` (at release specifically)

**Transverse (rotational) thorax velocity — for reference, separate from Cyl 3 headline:**
- `Thorax_Ang_Vel_max` (peak thorax angular velocity)

**Forward trunk flex at release (Cyl 3 partner):**
- `Trunk_Angle@Release_X`

**Cyl 3 product:** `MaxTrunkLinearVel_MPH_Y × Trunk_Angle@Release_X`

### Cylinder 4 — Front Leg Block
- `Lead_Knee_Angle@Footstrike_X` (flexion at landing)
- `Lead_Knee_Angle@Release_X` (flexion at release)
- `Lead_Knee_Angle@Max_Shoulder_Rot_X`
- Knee extension = FS − Release
- `Lead GRF_BW@Release_Z` (vertical force at release, in BW)
- `Lead GRF_BW@Footstrike_Z` (vertical force at landing)

### Cylinder 5 — Trunk Tilt at Release
- `Trunk_Angle@Release_X` (forward flex)
- `Trunk_Angle@Release_Y` (lateral tilt)
- `Trunk_Angle@Release_Z` (rotation finish)

### Cylinder 6 — Horizontal Abduction (Arm Trail) — CRITICAL TO READ X AXIS NOT Y
- `Pitching_Shoulder_Angle@Footstrike_X` (snapshot at FS)
- `Pitching_Shoulder_Angle@Footstrike_<10..120>ms_X` (post-FS progression for dwell time)
- `Pitching_Shoulder_Angle@Max_Shoulder_Rot_X` (at MER)

**Dwell time calculation:** Find the largest time-after-FS at which X is still ≤ −20°. That's the dwell.

**Do NOT use** `Pitching_Shoulder_Angle@Footstrike_Y` for Cyl 6. Y is vertical abduction (a different dimension).

### Arm timing (related to Cyl 6 but separate)
- `Pitching_Shoulder_Angle@Footstrike_Z` (external rotation at FS — arm timing)
- `Pitching_Shoulder_Angle@Max_Shoulder_Rot_Z` (max ER / MER)

### Cylinder 7 — Drive Leg / GRF
- `Back_Leg_GRF_mag_max` (back leg peak GRF, BW)
- `Lead GRF_BW@Footstrike_Y` (lead leg AP braking force, negative = braking)

### Cylinder 8 — Arm Health
- `Elbow_Torque_Nm@Max_Shoulder_Rot` (elbow varus torque at MER, Nm)
- `Elbow_Torque_Nm@Release`
- `Max_Elbow_Force` (peak elbow force, BW)
- `Shoulder_Force_Abd_N@Max_Shoulder_Abd_Force_N` (shoulder distraction, N)
- `Shoulder_Force_Abd@Max_Shoulder_Abd_Force` (shoulder distraction, BW)

## Kinematic sequence fields (for chain analysis, separate from 8 cylinders)

### Peak segment angular velocities
- `Pelvis_Ang_Vel_max`
- `Thorax_Ang_Vel_max`
- `Pitching_Humerus_Ang_Vel_max`
- `Pitching_Shoulder_Ang_Vel_max`
- `Pitching_Elbow_Ang_Vel_max`
- `Pitching_Hand_Ang_Vel_max`

### Peak timing (relative to start of capture)
- `MaxPelvisVelTime`
- `MaxThoraxVelTime`
- `MaxHumerusVelTime`
- `MaxHandVelTime`
- `FootstrikeTime` (subtract to get relative timing)
- `ReleaseTime`
- `Max_Shoulder_RotTime`
- `PelvisRot_StopTime` (for pelvis stop mechanism)

### Linear velocities (mph)
- `MaxPelvisLinearVel_MPH_X` (lateral)
- `MaxPelvisLinearVel_MPH_Y` (forward toward home — sagittal)
- `MaxPelvisLinearVel_MPH_Z` (vertical)
- `MaxTrunkLinearVel_MPH_X`
- `MaxTrunkLinearVel_MPH_Y`
- `MaxTrunkLinearVel_MPH_Z`

### Pelvis deceleration mechanism
- `Pelvis_Ang_Vel@MaxKneeHeight_X` (rotational velocity at top of leg lift)
- `Pelvis_Ang_Vel@Footstrike_X` (at landing — key for "is pelvis loaded?")
- `Pelvis_Ang_Vel@Max_Shoulder_Rot_X` (at MER — should be much lower if decel is working)
- `Back_Knee_Angle@MaxKneeHeight_X`, `@Footstrike_X`, `@Max_Shoulder_Rot_X`, `@Release_X`
  - The back knee FLEXES during the FS→MER window as it absorbs the pelvis deceleration

## Data structure of a B_Young report

A report contains:
- `results`: array of metric records. Each record has:
  - `id`: the metric name (e.g., `STRIDE_LENGTH_MEAN_PERCENT`)
  - `type`: `scalar` or `series` (scalars are single values per pitch; series are time-series data)
  - `path`: organizing category (e.g., `METRIC/PROCESSED`, `METRIC/STRIDE_LENGTH`)
  - `data`: array of `{measurement: "Fastball RH 7", values: [82.3]}` per pitch
- `events`: phase boundary timings per pitch
- `measurements`: per-pitch metadata including velocity in the `Comments` field
- `subject`: pitcher identity
- `project`: handedness (e.g., "Baseball Right-handed")
- `metadata.customFields`: DOB, height, weight, etc.

## Extracting velocity from a pitch

Velocity is stored in the `Comments` field of each Fastball measurement:

```python
for m in metadata['measurements']:
    if 'Fastball' in m['id']:
        fields = {f['id']: f.get('value') for f in m.get('fields', [])}
        velocity_mph = float(fields.get('Comments', '0'))
```

The Visual3D output has all `SPIN_RATE` and `SPIN_AXIS` set to 0 in our captures (the system doesn't measure those). Use the Comments field for ball velocity.