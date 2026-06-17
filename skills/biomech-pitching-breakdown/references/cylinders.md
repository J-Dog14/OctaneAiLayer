# The 8 Cylinders — Detailed Definitions

The 8ctane Baseball framework organizes a pitcher's biomechanical profile into eight mechanical categories. Each cylinder is named, has specific metric(s), specific benchmark ranges, and specific coaching implications.

## Cylinder 1 — Stride

**What it measures:** How far the lead foot travels from the rubber, normalized to standing height.

**Headline metric:** `STRIDE_LENGTH_MEAN_PERCENT`

**Elite range:** 85–90% of height

**Why it matters:** Stride is the foundation. A short stride compresses the time the chain has to develop, limits how far the pelvis can rotate, and shortens release-point reach. Most velocity gains for shorter-strided pitchers come once stride lengthens.

**Cohort observations:**
- Trevor Cleveland (93%) is the highest in the cohort
- Carson Carroll (89%) is at the top of elite range
- Isaac James (68%) is well below — the shortest in cohort
- Most pitchers cluster 75–82%

## Cylinder 2 — Hip–Shoulder Separation

**What it measures:** Trunk rotation relative to pelvis at footstrike — the "loaded spring" of the throw.

**Headline metric:** `Trunk Rot wrt Pelvis Rot@Footstrike`

**Elite range:** 40–60°

**Why it matters:** HSS at FS is the stretch loaded between the lower body (already opening) and the upper body (still closed). When the pelvis fires, the trunk springs through that stretch. Bigger HSS = bigger spring = more amplification through the chain.

**Cohort observations:**
- Isaac James (72°) — slightly hyper, possibly compensating for short stride
- Boyatt (63°), JT (59°), Wall (60°), Eli (57°), Fletcher (57°) — strong loading
- Stacks Dylan (37°) — under-loaded despite being a 91.7 mph thrower (uses different mechanism)
- Luke Johnson (14°) — major flag, chain not loaded

## Cylinder 3 — Trunk Velocity (sagittal × transverse)

**What it measures:** Sagittal-plane trunk velocity (moving DOWN the mound, toward home) PAIRED WITH trunk forward flex at release. The product captures the pairing.

**Headline metrics:**
- Sagittal: `MaxTrunkLinearVel_MPH_Y`
- Forward flex: `Trunk_Angle@Release_X`
- Product: sagittal × forward flex

**Elite range:** Pair both high. Product range in cohort: 249–472 (excluding outliers).

**Why it matters:** These are two independent dimensions (cohort r = 0.06 between them) that both contribute to velocity. A pitcher with high sagittal but low flex (Patrick Maclean) has the trunk moving but doesn't dump it; a pitcher with high flex but low sagittal (Carson Carroll) is compensating with extreme dump. Pair high = balanced engine. Either alone leaves velocity on the table.

**Critical:** This is NOT the kinematic sequence (peak segment velocities). It's specifically about trunk delivery.

**Cohort observations:**
- Carson Carroll (472) — extreme flex compensating for under-amplified rotation
- Dylan Stacks (433) — balanced both
- Boyatt (357) — balanced
- McLamb (378) — highest sagittal in cohort
- Luke Johnson (7) — outlier flag; only 1° forward flex at release

## Cylinder 4 — Front Leg Block

**What it measures:** Lead-knee extension from footstrike to release, plus lead-leg vertical GRF at release.

**Headline metrics:**
- Knee extension: `Lead_Knee_Angle@Footstrike_X − Lead_Knee_Angle@Release_X`
- Vertical GRF: `Lead GRF_BW@Release_Z`

**Elite range:** 15–25° knee extension, 2.0+ BW vertical GRF

**Why it matters:** The lead leg posts up to redirect rotational energy from the lower body INTO the trunk. A stiff front leg = clean energy transfer. A soft block (knee flexing during the throw) = energy leak.

**Cohort observations:**
- Isaac James (58°) — extreme, knee actually hyperextends
- Parnell (32°), Fletcher (27°), Wall (26°) — elite block
- Wall (2.35 BW) — highest vertical GRF in cohort
- Boyatt (13°), Patrick (8°), McLamb (10°), JT (8°), Carson Carroll (7°) — softer blocks

## Cylinder 5 — Trunk Tilt at Release

**What it measures:** Forward flex AND lateral flex at ball release. Both dimensions.

**Headline metrics:**
- Forward: `Trunk_Angle@Release_X`
- Lateral: `Trunk_Angle@Release_Y`

**Elite ranges:** 30–40° forward, 30–45° lateral

**Why it matters:** Forward flex provides downhill plane and release extension. Lateral tilt provides side-bend that lets the arm slot deliver from a height. Both contribute to velocity and release point.

**Cohort observations:**
- Carson Carroll (59° forward) — aggressive dump
- Stacks (50° / 42°) — both aggressive
- Patrick (28°) — leaves forward flex on the table
- Connor Wong (6° lateral) — almost no side bend, nearly vertical finish
- Carson Crider (15° forward) — second-most upright finish in cohort

## Cylinder 6 — Horizontal Abduction (Arm Trail)

**What it measures:** How far the humerus stays trailing behind the midline of the body. Negative values = arm behind shoulders' line (good trail).

**Headline metrics:**
- `Pitching_Shoulder_Angle@Footstrike_X` (value at FS)
- Progression at FS+10ms, +20ms, ... +120ms (dwell time)
- "Dwell ≤ −20°" = how long the angle stays significantly trailing

**Coaching cue:** Hard throwers HOLD horizontal abduction LONGER. Dwell time is the key metric, not just the snapshot.

**Critical:** This is the X axis, NOT the Y axis. Y is vertical abduction (arm raised relative to body) — a different dimension. A pitcher can have high Y (105°+ vertical) and still have healthy X (good trail). Do not flag pitchers for "high cock" based on Y readings.

**Cohort observations:**
- Cyl 6 dwell correlation with velocity: r = +0.38 (confirms framework)
- JT Williams (80 ms dwell) — longest in cohort, arm trail is a strength (not a flag)
- Carson Carroll (−65° at FS) — most arm trail in cohort
- Trevor Cleveland (20 ms dwell) — shortest dwell, releases arm quickly
- Patrick Maclean (−28° at FS) — least arm trail, most coachable area

## Cylinder 7 — Drive Leg / GRF

**What it measures:** Back-leg peak ground reaction force (drive engine) plus lead-leg anteroposterior braking force at footstrike.

**Headline metrics:**
- Back-leg GRF: `Back_Leg_GRF_mag_max`
- Lead-leg braking: `Lead GRF_BW@Footstrike_Y` (negative = braking)

**Elite ranges:** Back leg 1.6–1.8 BW; braking > 1.0 BW

**Why it matters:** The drive leg generates the lateral momentum that the front leg has to brake/redirect. A weak drive leg = a quiet pelvis; weak braking = energy lost on landing.

**Cohort observations:**
- Parnell (1.89 BW back, −0.66 brake) — most powerful drive
- Boyatt (1.61 / −0.95) — balanced
- Stacks (1.78 / −1.14) — strongest braking in cohort
- JT (1.23 / −0.69) — weak both
- Johnson (1.62 / −0.92) — solid despite young age

## Cylinder 8 — Arm Health

**What it measures:** Elbow varus torque at maximum external rotation (UCL stress measure) and shoulder distraction force at deceleration.

**Headline metrics:**
- Elbow torque: `Elbow_Torque_Nm@Max_Shoulder_Rot`
- Shoulder distraction: `Shoulder_Force_Abd_N@Max_Shoulder_Abd_Force_N`, `Shoulder_Force_Abd@Max_Shoulder_Abd_Force` (BW)

**Reference ranges:** Elbow varus torque 80–120 Nm (>130 = caution). Shoulder distraction 0.8–1.1 BW at game effort (lower = sub-max capture).

**Why it matters:** These are the cost the arm is paying for the velocity. High torque + low velocity = inefficient delivery, arm rescuing the chain.

**Stress-per-mph framing:** Normalize by velocity: `elbow_torque / velocity`. Elite arms run 0.9–1.2 Nm/mph; elevated-stress arms run 1.4+. This is more diagnostic than absolute torque.

**Cohort observations:**
- JT Williams (127 Nm / 86.8 mph = 1.46) — top of stress-per-mph
- McLamb (126 / 85.8 = 1.47), Carson Crider (118 / 85.2 = 1.39), Stacks (125 / 91.7 = 1.36) — elevated
- Boyatt (115 / 93.2 = 1.23), Trevor (94 / 81.2 = 1.16), Isaac (87 / 80.9 = 1.08) — healthy
- Most shoulder distractions in cohort 0.4–0.6 BW (sub-max captures)

## How to read across cylinders

A clean profile has most cylinders in the "Strength" or "Elite" band. A pitcher with a single big leak usually has it concentrated in one or two cylinders. Common patterns:

- **"Arm Thrower":** Low Cyl 4 (soft block), low Cyl 7 (weak drive), high Cyl 8 stress per mph. Arm doing too much. Example: JT Williams
- **"Sub-max effort capture":** Cyl 8 shoulder distraction below 0.5 BW. Likely needs re-test at game effort. Example: Trevor Cleveland
- **"Short stride compensator":** Cyl 1 short (<75%), Cyl 2 hyper-separated, Cyl 5 trunk dumps to compensate. Example: Isaac James
- **"Quiet trunk":** Cyl 3 product low, Cyl 5 forward flex low, arm rescuing chain. Example: Carson Crider, Luke Johnson
- **"Model arm":** All cylinders Strength+, Cyl 8 healthy stress/mph. Examples: Boyatt, Stacks (with one minor leak each)