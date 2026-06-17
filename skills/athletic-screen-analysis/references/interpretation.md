# Assessment Interpretation Guide

## Drop Jump (DJ)

**Protocol**
Three trials. Trials 1–2 are uncoached. Trial 3 is cued — default cue is "be as quick off
the ground as possible." If the athlete is already very quick but height suffers, cue is
flipped to "take longer but give me your best vertical." Use the best trial.
Arms are permitted — this is a whole-body coordination test, not isolated lower body.

**Why it's weighted most heavily**
The DJ mimics the front-leg plant in pitching: sudden ground contact → brace → extend,
all in a split second. It also reveals whole-body athleticism because arms are allowed.

**RSI target: ≥ 3.0** (database-stored value, already ×2). RSI near 80th percentile or above.
**CT target: ≤ 0.33s** (≈ 10th–15th percentile — remember: lower percentile = better).

**Decomposing RSI**
- Short CT + high jump = genuinely reactive. The ideal. Front leg reacts and launches fast.
- Long CT + very high jump = powerful but not reactive. Gets the height the hard way.
- Short CT + low jump = quick but not powerful. Needs strength.

**Force-velocity in the DJ**
Force-dominant (force_at_pp high, vel_at_pp lower): Strong at ground contact but slow to
transition through it. Front leg can absorb force but doesn't redirect it fast enough.
Velocity-dominant (vel_at_pp high, force_at_pp lower): Quick transition but lacks force base.
Balanced: ideal — both gauges in the upper distribution.

**Power curve priorities for DJ**
- `rpd_max_w_per_s`: rate of power ramp — critical for reactive movements. 90th percentile ≈ 31,000 W/s.
- `auc_j`: sustained work. 90th percentile ≈ 541 J. Below the 50th percentile (327 J) is a clear gap.
- `kurtosis`: ideal range is roughly the 30th–70th percentile of the DJ kurtosis distribution
  (-1.38 to -1.17). Too negative (very flat curve) = power isn't peaking sharply enough.
  Too positive (peaked) = power spikes and drops immediately.
- `fwhm_s`: full width at half maximum — how long the athlete is "in" their peak power window.
  Wider is generally better; narrow fwhm with low AUC = spike-and-drop pattern.
- `work_early_pct`: percentage of total work done in the first half of the movement. Very high
  (>85%) means the athlete does most work early and drops off — AUC will be low. Balanced
  work distribution correlates with better sustained output.

---

## Counter Movement Jump (CMJ)

**Protocol**
Hands on hips, flat-footed start. Load into countermovement and jump as high as possible.
Pure lower-body isolation. No arms.

**Purpose**
The CMJ is the isolated lower-body baseline. Compare CMJ to DJ: if DJ jump height
(with arms) is only slightly higher than CMJ, the athlete isn't leveraging upper-body
coordination. If DJ is significantly higher, the whole-body integration is working.

**Force-velocity in the CMJ**
The CMJ commonly shows the most direct picture of drive-leg quality. Force-dominant profile
in CMJ = strength is there but drive-leg speed is lacking. Velocity-dominant = quick off the
rubber but lacks the force base. Compare to the DJ profile — consistent pattern across both
confirms a training disposition rather than a movement artifact.

**Power curve priorities for CMJ**
- `rpd_max_w_per_s`: 85th percentile ≈ 21,500 W/s; 90th ≈ 26,800 W/s.
- `auc_j`: 90th percentile ≈ 1,200 J. The CMJ typically has the highest absolute AUC of all
  four assessments. A large AUC gap here (below 50th, ≈ 561 J) indicates the lower half
  cannot sustain force through full hip extension.
- `kurtosis`: The CMJ kurtosis distribution is notably wider than the DJ. The 50th percentile
  is ≈ -0.88. Values close to or above 0 indicate a very peaked curve (power spikes and
  collapses). Values near -1.5 are very flat. Ideal is roughly between the 25th and 60th
  percentile (-1.21 to -0.46).
- `time_to_rpd_max_s`: When in the movement peak rate of power development occurs. Late
  RPD (high percentile, >1.0s) means the athlete accelerates hard but starts slow. Early RPD
  (low percentile, <0.4s) means they ramp fast — favorable for pitching.

---

## Plyo Pushup (PPU)

**Protocol**
Explosive pushup from force plate. The athlete claps or just explodes off the plate as high
as possible. `jh_in` = hand height off the plate (upper-body "jump height").

**Why it matters**
Only direct upper-body explosive assessment in the screen. Most athletes show a force-dominant
profile here because they train sets/reps (bench, rows) but not ballistic upper-body work.
Athletes who specifically train upper-body explosiveness (med ball throws, ballistic pressing)
stand out. This is where athletes get humbled OR validate their training approach.

**The typical pattern**
Force-dominant (force_at_pp high, vel_at_pp lower) is the norm — especially in bigger,
stronger athletes. High force at PP but low velocity = the upper half is strong but slow.
The power curve in the PPU is often irregular — jagged mean trace, inconsistent neuromuscular
recruitment — because ballistic upper-body training is rare. Note irregularity when you see it.

**Force-velocity in the PPU**
A velocity-dominant or balanced profile on the PPU is uncommon and meaningful — it signals
the athlete has done deliberate explosive upper-body training. Flag this positively.

**Power curve priorities for PPU**
- `rpd_max_w_per_s`: 85th percentile ≈ 8,400 W/s; 90th ≈ 10,600 W/s. Reaching the 85th
  percentile here is a real indicator of upper-body explosive quality.
- `auc_j`: 90th percentile ≈ 626 J; 50th ≈ 326 J. Upper-body AUC is naturally lower than
  lower-body. Low AUC relative to high peak power = power is a brief spike, not sustained.
- `kurtosis`: The PPU kurtosis distribution is skewed very negative — the 50th percentile is
  -1.45. An ideal curve for the PPU sits in the -1.6 to -1.3 range (roughly 30th–70th
  percentile). Values above -0.9 (>90th percentile) = too peaked for this movement.
- `jh_in` vs `pp_w_per_kg`: A large gap between these percentiles (high power, low height)
  means the athlete generates force but doesn't express it through range of motion.

**Connecting PPU to pitching**
The upper half receives energy from the hip-trunk complex and accelerates the arm. An upper
half that produces force slowly (low RPD, force-dominant) is a kinetic chain bottleneck
regardless of raw strength. Arm acceleration is a high-velocity event.

---

## Single Leg Vertical (SLV)

**Protocol**
One leg at a time. Arms allowed — same philosophy as DJ: assess whole-athlete coordination
on a single leg, not artificial isolation. Best trial per leg.

**Left vs. Right for pitchers**
For a right-handed pitcher: left leg = plant/front leg; right leg = drive leg.
For a left-handed pitcher: reverse this.

**Plant leg should outperform drive leg — this is expected and desirable.**
The plant leg hits the mound and reacts to ground contact hundreds of times per week across
games, bullpen sessions, and catch play. This repeated reactive load naturally develops the
plant leg's explosiveness and stability. It is sport-specific adaptation, not a training
asymmetry. Do not flag plant leg > drive leg as a problem.

**Training philosophy:** Both legs are trained identically. The goal is for both to improve
over time. If the drive leg never catches the plant leg, that is acceptable. Over-emphasizing
the drive leg to "catch up" would either overtrain it or neglect the plant leg — both wrong.

**When to monitor asymmetry:** If the drive leg shows very large absolute gaps in force or
AUC (not just percentile) compared to the plant leg, note it as a baseline to track — but
frame it as monitoring, not a deficit requiring correction.

**Power curve priorities for SLV**
- `rpd_max_w_per_s`: 85th percentile ≈ 19,200 W/s (left), 17,300 W/s (right). Reaching
  the 80th+ percentile on both legs for RPD is a strong indicator of single-leg explosiveness.
- `auc_j`: 85th percentile ≈ 1,119 J (left), 1,149 J (right). Elite single-leg AUC means
  the athlete can sustain force through the full single-leg jump — directly relevant to
  both plant-leg bracing and drive-leg push-off.
- `kurtosis`: The SLV kurtosis distribution has a higher (less negative) mean than other
  assessments. The 50th percentile is approximately -0.45 (left) and -0.52 (right). Values
  above +0.5 = power very concentrated in a narrow window. For single-leg explosive quality
  in pitching, a broad-to-moderate curve (negative kurtosis) is preferable.
- Side-by-side radar comparison: When both legs are reported, compare the radar shapes. If
  one leg has high RPD but low AUC and the other has the opposite pattern, they are producing
  power through different strategies — worth noting in the coaching context.

---

## Power Curve General Guide

### rpd_max_w_per_s (Max Rate of Power Development)
How fast power ramps up. This is often more important than peak power for sport-speed
events — the movement ends before a slow ramp reaches its ceiling.

### kurtosis
Shape of the power-time distribution:
- Very negative (< -1.5): Flat, distributed curve. Power spreads broadly over time.
  Can mean slow overall ramp or good sustained output. Check AUC to distinguish.
- Moderate negative (-1.5 to -0.5): Ideal for most movements. Clear peak, reasonable width.
- Near zero or positive (> -0.3): Sharp, narrow peak. Power spikes and drops immediately.
  Almost always associated with low AUC. The athlete "peaks and leaves" the movement.

### auc_j (Work / Area Under the Curve)
Total impulse delivered. Low AUC + high peak power = the athlete spikes but doesn't stay.
This is the most common "hidden" gap — athletes look explosive in peak metrics but the
sustained application is weak. In pitching this matters because:
- Front leg must maintain stiffness through ball release (DJ, SLV left AUC)
- Drive leg must extend fully through hip rotation (CMJ, SLV right AUC)
- Upper half must push through arm acceleration (PPU AUC)

### rise_time_10_90_s
How long it takes to go from 10% to 90% of peak power. Shorter = faster ramp.
Compare this to time_to_rpd_max_s: both should be early in the movement for reactive tests.

### fwhm_s (Full Width at Half Maximum)
Width of the power curve at 50% of peak. Wider = more time spent near peak power.
Narrower with low AUC = the peak is brief and steep. Use as confirmation alongside kurtosis.

### work_early_pct
Percentage of total work done in the first half of the movement. A very high value (>85%)
paired with low AUC signals front-loading: the athlete generates most of their work early
and drops off before completing the movement. This is common in athletes with poor
sustained force application and usually visible as a steep rise followed by a sharp fall
in the power curve.

---

## Cross-Movement Pattern Recognition

**CMJ ≈ DJ in jump height percentile (arms don't add much)**
Upper-body coordination gap. The athlete generates power bilaterally but can't amplify it
with arm-driven whole-body momentum. Or the reactive stretch-shortening cycle is limiting
the DJ. Compare RSI and CT for context.

**Consistently low AUC across DJ + CMJ**
Global bilateral sustained-force gap. Not assessment-specific — a training pattern issue.
The athlete spikes to power but doesn't stay in it. Most directly relevant to kinetic chain
continuity through ball release.

**DJ AUC low, SLV AUC elite**
Interesting pattern: can sustain single-leg force but not bilateral reactive force. May
indicate the bilateral reactive context disrupts their sustained output, or that the reactive
speed requirement interrupts the loading-to-sustain pattern.

**PPU force-dominant, CMJ/DJ balanced**
Classic strength-trained athlete. Lower half training has included enough plyometric work
to balance the F-V curve; upper half training has not. Upper-body speed-strength is the gap.

**High peak power everywhere, low AUC everywhere**
Explosive but brief athlete. Every movement. This is a global neuromuscular pattern —
the athlete always peaks and retreats. Training that prolongs the high-power window
(contrast methods, full-ROM explosive movements) is the long-term direction.