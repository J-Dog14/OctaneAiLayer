"""
Warehouse metric key → coach-readable language.

    from src.research.metric_display import DISPLAY

    DISPLAY.name("fm_lead_rfd_braking_bw_per_s")
    # 'Front-leg braking rate'
    DISPLAY.definition("fm_lead_rfd_braking_bw_per_s")
    # 'How fast the front leg gets the brakes on after the foot lands.'
    DISPLAY.format_value("fm_lead_rfd_braking_bw_per_s", 18.57)
    # '18.6 BW/s'
    DISPLAY.describe_change("fm_lead_rfd_braking_bw_per_s", +2.4)
    # ('improved', '+2.4 BW/s')

Three tiers of coverage, in order of preference:

  curated  — an entry in metric_display.yaml written by a human. Only these
             are allowed on a coach-facing page with plain language.
  auto     — composed from the vocabulary tables (e.g. 'pelvis rotation speed
             at ball release, axis Z'). Readable, but nobody has signed off,
             so it is analyst-appendix only.
  raw      — no rule matched. The key is cleaned up cosmetically and shown
             as-is.

`python -m src.main research metric-coverage` prints what is still uncurated,
ranked by how often each key actually appears in analyses, so the naming
backlog is worked in impact order rather than alphabetically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import yaml

_YAML_PATH = Path(__file__).resolve().parent / "references" / "metric_display.yaml"

# direction vocabulary
HIGHER_BETTER = "higher_better"
LOWER_BETTER = "lower_better"
CONTEXT = "context"
LOAD = "load"
REVIEW = "review"

_SIGNED_DIRECTIONS = {HIGHER_BETTER, LOWER_BETTER}

STATUS_CURATED = "curated"
STATUS_AUTO = "auto"
STATUS_RAW = "raw"


@dataclass(frozen=True)
class MetricDisplay:
    key: str
    name: str
    short: str
    definition: str
    unit: str
    direction: str
    group: str
    status: str
    cue: str | None = None

    @property
    def is_coach_ready(self) -> bool:
        return self.status == STATUS_CURATED

    @property
    def has_direction(self) -> bool:
        """True when 'better' is a meaningful word for this metric."""
        return self.direction in _SIGNED_DIRECTIONS

    def with_unit(self, value: float | None, *, precision: int | None = None) -> str:
        if value is None:
            return "—"
        p = precision if precision is not None else _auto_precision(value)
        txt = f"{value:,.{p}f}"
        return f"{txt} {self.unit}".strip()


def _auto_precision(value: float) -> int:
    a = abs(value)
    if a == 0:
        return 1
    if a >= 1000:
        return 0
    if a >= 100:
        return 1
    if a >= 1:
        return 2
    if a >= 0.01:
        return 3
    return 4


# ──────────────────────────────────────────────────────────────────────────
# Composer — builds an 'auto' entry for uncurated structured keys
# ──────────────────────────────────────────────────────────────────────────

_KIN_PROCESSED = re.compile(
    r"^kin_PROCESSED\.(?P<body>.+?)(?:@(?P<event>[A-Za-z0-9_.]+))?\.(?P<axis>[XYZ])$"
)
_KIN_INCREMENT = re.compile(
    r"^kin_INCREMENT\.(?P<body>.+?)@(?P<event>[A-Za-z0-9_]+?)_(?P<offset>[0-9.]+)ms\.(?P<axis>[XYZ])$"
)
_KIN_TIMING = re.compile(r"^kin_TIMING\.(?P<body>.+?)\.(?P<axis>[XYZ])$")
_KIN_SEQ = re.compile(r"^kin_KINEMATIC_SEQUENCE\.(?P<body>.+?)\.(?P<axis>[XYZ])$")


class _Composer:
    def __init__(self, vocab: dict[str, Any]):
        self.segments: dict[str, str] = vocab.get("segments", {}) or {}
        self.quantities: dict[str, dict] = vocab.get("quantities", {}) or {}
        self.events: dict[str, str] = vocab.get("events", {}) or {}
        self.axis_labels: dict[str, str] = vocab.get("axis_labels", {}) or {}
        self.fm_sides: dict[str, str] = vocab.get("fm_sides", {}) or {}
        self.fm_measures: dict[str, str] = vocab.get("fm_measures", {}) or {}
        self.fm_windows: dict[str, str] = vocab.get("fm_windows", {}) or {}
        self.fm_units: dict[str, str] = vocab.get("fm_units", {}) or {}
        # Longest-first so 'Pitching_Shoulder' wins over 'Shoulder'.
        self._segments_by_len = sorted(self.segments, key=len, reverse=True)
        self._quantities_by_len = sorted(self.quantities, key=len, reverse=True)
        self._fm_measures_by_len = sorted(self.fm_measures, key=len, reverse=True)
        self._fm_windows_by_len = sorted(self.fm_windows, key=len, reverse=True)
        self._fm_units_by_len = sorted(self.fm_units, key=len, reverse=True)

    # ── kin_* ─────────────────────────────────────────────────────────────

    def _split_body(self, body: str) -> tuple[str | None, str | None, str]:
        """'Pelvis_Ang_Vel' -> ('Pelvis', 'Ang_Vel', ''). Remainder is anything
        we could not attribute.

        Visual3D names are not consistently `<Segment>_<Quantity>`: plenty look
        like `Max_Elbow_Varus_Torque_Nm` or `Shoulder_Rotation_Cummulative_
        Range_FStoRelease`. So after trying the strict prefix split we fall
        back to locating a segment and a quantity token *anywhere* in the
        string, which is what makes the long tail come out readable instead of
        as 'Value (max elbow varus torque nm)'.
        """
        seg = next((s for s in self._segments_by_len if body.startswith(s)), None)
        rest = body[len(seg):].lstrip("_") if seg else body
        qty = next((q for q in self._quantities_by_len if rest == q or rest.startswith(q)), None)
        if qty is not None:
            return seg, qty, rest[len(qty):].lstrip("_")

        # Fallback: token-anywhere search.
        if seg is None:
            seg = next((s for s in self._segments_by_len
                        if re.search(rf"(?:^|_){re.escape(s)}(?:_|$)", body)), None)
        search_space = body
        if seg:
            search_space = re.sub(rf"(?:^|_){re.escape(seg)}(?=_|$)", "", body).strip("_")
        qty = next((q for q in self._quantities_by_len
                    if re.search(rf"(?:^|_){re.escape(q)}(?:_|$)", search_space)), None)
        if qty is None:
            return seg, None, search_space
        remainder = re.sub(rf"(?:^|_){re.escape(qty)}(?=_|$)", "", search_space).strip("_")
        return seg, qty, remainder

    def _axis_phrase(self, seg: str | None, qty: str | None, axis: str) -> str:
        explicit = self.axis_labels.get(f"{seg}.{qty}.{axis}") if seg and qty else None
        if explicit:
            return explicit
        return f"axis {axis}"

    def compose_kin(self, key: str) -> MetricDisplay | None:
        m = _KIN_PROCESSED.match(key)
        if m:
            seg, qty, remainder = self._split_body(m.group("body"))
            event = m.group("event")
            return self._build_kin(key, seg, qty, remainder, event, m.group("axis"))

        m = _KIN_INCREMENT.match(key)
        if m:
            seg, qty, remainder = self._split_body(m.group("body"))
            event = m.group("event")
            base = self._build_kin(key, seg, qty, remainder, event, m.group("axis"))
            off = m.group("offset")
            return _replace_display(
                base,
                name=f"{base.name}, {off} ms window",
                definition=(f"{base.definition} Sampled in the {off} ms window "
                            f"after {self.events.get(event, _humanize(event))}."),
            )

        m = _KIN_TIMING.match(key)
        if m:
            body = m.group("body")
            label = _humanize(re.sub(r"Time$", "", body))
            return MetricDisplay(
                key=key,
                name=f"Time of {label.lower()}",
                short=f"t({label})",
                definition=f"When {label.lower()} happens within the delivery.",
                unit="s",
                direction=CONTEXT,
                group="timing",
                status=STATUS_AUTO,
            )

        m = _KIN_SEQ.match(key)
        if m:
            seg, qty, _ = self._split_body(re.sub(r"_max$", "", m.group("body")))
            seg_lbl = self.segments.get(seg, _humanize(seg or "segment"))
            return MetricDisplay(
                key=key,
                name=f"Peak {seg_lbl} rotation speed (kinematic sequence)",
                short=f"Peak {seg_lbl} speed",
                definition=f"Top rotation speed of the {seg_lbl} in the kinematic sequence.",
                unit="deg/s",
                direction=HIGHER_BETTER,
                group="sequencing",
                status=STATUS_AUTO,
            )
        return None

    def _build_kin(self, key, seg, qty, remainder, event, axis) -> MetricDisplay:
        seg_lbl = self.segments.get(seg, _humanize(seg) if seg else "")
        q = self.quantities.get(qty, {}) if qty else {}
        qty_lbl = q.get("label", _humanize(qty) if qty else "value")
        unit = q.get("unit", "")
        direction = q.get("direction", CONTEXT)
        event_lbl = self.events.get(event, _humanize(event)) if event else None

        if qty is None:
            # No known quantity token — the remainder IS the descriptor, so
            # read it out rather than emitting a useless 'Value (...)'.
            core = " ".join(p for p in (seg_lbl, _humanize(remainder).lower()) if p)
            core = core or _humanize(key)
        else:
            core = " ".join(p for p in (seg_lbl, qty_lbl) if p) or _humanize(key)
            if remainder:
                core = f"{core} ({_humanize(remainder).lower()})"
        name = core[:1].upper() + core[1:]
        if event_lbl:
            name = f"{name} at {event_lbl}"
        axis_phrase = self._axis_phrase(seg, qty, axis)
        name = f"{name} — {axis_phrase}"

        definition = (
            f"{qty_lbl.capitalize()} of the {seg_lbl or 'segment'}"
            + (f" at {event_lbl}" if event_lbl else "")
            + f", measured on {axis_phrase}."
        )
        group = _kin_group(qty, seg)
        return MetricDisplay(
            key=key, name=name,
            short=f"{seg_lbl or _humanize(key)} {qty_lbl}".strip()[:38],
            definition=definition, unit=unit, direction=direction,
            group=group, status=STATUS_AUTO,
        )

    # ── fm_* ──────────────────────────────────────────────────────────────

    def compose_fm(self, key: str) -> MetricDisplay | None:
        if not key.startswith("fm_"):
            return None
        rest = key[3:]
        side = next((s for s in self.fm_sides if rest.startswith(s + "_")), None)
        if side:
            rest = rest[len(side) + 1:]
        unit_token = next(
            (u for u in self._fm_units_by_len if rest == u or rest.endswith("_" + u)), None
        )
        if unit_token and rest != unit_token:
            rest = rest[: -(len(unit_token) + 1)]
        measure = next(
            (m for m in self._fm_measures_by_len if rest == m or rest.startswith(m + "_")), None
        )
        if measure:
            rest = rest[len(measure):].lstrip("_")
        window = next(
            (w for w in self._fm_windows_by_len if rest == w or rest.startswith(w)), None
        )
        if not (side or measure):
            return None

        side_lbl = self.fm_sides.get(side, "")
        measure_lbl = self.fm_measures.get(measure, _humanize(measure or rest).lower())
        window_lbl = self.fm_windows.get(window, "")
        unit = self.fm_units.get(unit_token, "")

        name = " ".join(p for p in (side_lbl.capitalize() if side_lbl else "",
                                    measure_lbl) if p).strip()
        if window_lbl:
            name = f"{name} {window_lbl}"
        name = name[:1].upper() + name[1:] if name else key
        definition = (
            f"Force-plate measure: {measure_lbl}"
            + (f" through the {side_lbl}" if side_lbl else "")
            + (f" {window_lbl}" if window_lbl else "")
            + "."
        )
        direction = HIGHER_BETTER if measure and (
            measure.startswith("peak") or measure.startswith("impulse")
            or measure.startswith("rfd")
        ) else CONTEXT
        return MetricDisplay(
            key=key, name=name, short=name[:38], definition=definition,
            unit=unit, direction=direction, group="force", status=STATUS_AUTO,
        )


def _kin_group(qty: str | None, seg: str | None) -> str:
    q = (qty or "").lower()
    s = (seg or "").lower()
    if "torque" in q or "dist_force" in q:
        return "load"
    if "grf" in q:
        return "force"
    if "ang_vel" in q or "ang_acc" in q:
        return "sequencing" if s in ("pelvis", "trunk", "thorax") else "arm"
    if "cog" in q:
        return "lower_half"
    if "knee" in s or "leg" in s:
        return "lower_half"
    if "shoulder" in s or "elbow" in s or "humerus" in s or "hand" in s:
        return "arm"
    return "posture"


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _humanize(token: str | None) -> str:
    """'MaxShoulderVelTime' -> 'Max Shoulder Vel Time'; 'Ang_Vel' -> 'Ang Vel'."""
    if not token:
        return ""
    t = token.replace("_", " ").replace(".", " ")
    t = " ".join(_CAMEL_BOUNDARY.sub(" ", part) for part in t.split())
    return re.sub(r"\s+", " ", t).strip()


def _replace_display(d: MetricDisplay, **kw) -> MetricDisplay:
    from dataclasses import replace as _r
    return _r(d, **kw)


# ──────────────────────────────────────────────────────────────────────────
# The registry
# ──────────────────────────────────────────────────────────────────────────

class MetricDisplayRegistry:
    def __init__(self, path: Path = _YAML_PATH):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self._groups: dict[str, dict] = raw.get("groups", {}) or {}
        self._vocab = raw.get("vocabulary", {}) or {}
        self._composer = _Composer(self._vocab)
        self._curated: dict[str, MetricDisplay] = {}
        self._aliases: dict[str, str] = {}

        entries = raw.get("metrics", {}) or {}
        for key, spec in entries.items():
            if spec and "alias_of" in spec:
                self._aliases[key] = spec["alias_of"]
                continue
            spec = spec or {}
            name = spec.get("name") or _humanize(key)
            self._curated[key] = MetricDisplay(
                key=key,
                name=name,
                short=spec.get("short") or name,
                definition=spec.get("definition", "").strip(),
                unit=str(spec.get("unit", "") or ""),
                direction=spec.get("direction", CONTEXT),
                group=spec.get("group", "other"),
                status=STATUS_CURATED,
                cue=spec.get("cue"),
            )
        # Resolve aliases now so lookups stay O(1) and cycles fail loudly.
        for alias, target in self._aliases.items():
            if target not in self._curated:
                raise ValueError(
                    f"metric_display.yaml: {alias!r} aliases {target!r}, which is "
                    f"not a curated metric."
                )

    # ── lookup ────────────────────────────────────────────────────────────

    @lru_cache(maxsize=4096)
    def get(self, key: str) -> MetricDisplay:
        if key in self._curated:
            return self._curated[key]
        if key in self._aliases:
            target = self._curated[self._aliases[key]]
            return _replace_display(target, key=key)
        composed = self._composer.compose_kin(key) or self._composer.compose_fm(key)
        if composed is not None:
            return composed
        cleaned = _humanize(re.sub(r"^(kin_|fm_|mob_|screen_|proteus_|pitch_|hit_|fp_|arm_|cb_|rs_)", "", key))
        pretty = cleaned[:1].upper() + cleaned[1:] if cleaned else key
        return MetricDisplay(
            key=key, name=pretty, short=pretty[:38], definition="",
            unit="", direction=CONTEXT, group="other", status=STATUS_RAW,
        )

    # convenience accessors
    def name(self, key: str) -> str: return self.get(key).name
    def short(self, key: str) -> str: return self.get(key).short
    def unit(self, key: str) -> str: return self.get(key).unit
    def definition(self, key: str) -> str: return self.get(key).definition
    def direction(self, key: str) -> str: return self.get(key).direction
    def cue(self, key: str) -> str | None: return self.get(key).cue
    def group(self, key: str) -> str: return self.get(key).group
    def is_coach_ready(self, key: str) -> bool: return self.get(key).is_coach_ready

    def group_label(self, group: str) -> str:
        return (self._groups.get(group) or {}).get("label", _humanize(group).title())

    def group_order(self, group: str) -> int:
        return (self._groups.get(group) or {}).get("order", 999)

    def format_value(self, key: str, value: float | None, *,
                     precision: int | None = None) -> str:
        return self.get(key).with_unit(value, precision=precision)

    def format_delta(self, key: str, delta: float | None, *,
                     precision: int | None = None) -> str:
        if delta is None:
            return "—"
        d = self.get(key)
        p = precision if precision is not None else _auto_precision(delta)
        return f"{delta:+,.{p}f} {d.unit}".strip()

    def describe_change(self, key: str, delta: float | None) -> tuple[str, str]:
        """(verb, formatted_delta). The verb respects the metric's direction:
        for a `lower_better` metric a negative delta reads as 'improved'.
        For `context` / `load` / `review` metrics we deliberately refuse to
        say better or worse — the word is 'increased' / 'decreased'."""
        if delta is None:
            return "unchanged", "—"
        txt = self.format_delta(key, delta)
        d = self.get(key)
        if abs(delta) == 0:
            return "unchanged", txt
        if d.direction == HIGHER_BETTER:
            return ("improved" if delta > 0 else "declined"), txt
        if d.direction == LOWER_BETTER:
            return ("improved" if delta < 0 else "declined"), txt
        return ("increased" if delta > 0 else "decreased"), txt

    def signed_toward_better(self, key: str, delta: float | None) -> int | None:
        """+1 the change moved toward 'better', -1 away, None if the metric
        has no agreed direction."""
        if delta is None or delta == 0:
            return None
        d = self.get(key)
        if d.direction == HIGHER_BETTER:
            return 1 if delta > 0 else -1
        if d.direction == LOWER_BETTER:
            return 1 if delta < 0 else -1
        return None

    # ── coverage ──────────────────────────────────────────────────────────

    def coverage(self, keys: Iterable[str],
                 weights: dict[str, int] | None = None):
        """Which keys are curated, auto-composed, or raw.

        weights: optional key → how often it appears, so the backlog can be
        worked in impact order instead of alphabetically.
        """
        import pandas as pd
        rows = []
        for k in sorted(set(keys)):
            d = self.get(k)
            rows.append({
                "metric": k,
                "status": d.status,
                "display_name": d.name,
                "unit": d.unit,
                "direction": d.direction,
                "group": d.group,
                "has_definition": bool(d.definition),
                "has_cue": bool(d.cue),
                "weight": (weights or {}).get(k, 0),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        order = {STATUS_RAW: 0, STATUS_AUTO: 1, STATUS_CURATED: 2}
        df["_o"] = df["status"].map(order)
        return (df.sort_values(["_o", "weight", "metric"],
                               ascending=[True, False, True])
                  .drop(columns=["_o"]).reset_index(drop=True))

    def needs_review(self, keys: Iterable[str]) -> list[str]:
        """Curated metrics whose direction was never decided."""
        return sorted(k for k in set(keys) if self.get(k).direction == REVIEW)


DISPLAY = MetricDisplayRegistry()

__all__ = [
    "DISPLAY", "MetricDisplay", "MetricDisplayRegistry",
    "HIGHER_BETTER", "LOWER_BETTER", "CONTEXT", "LOAD", "REVIEW",
    "STATUS_CURATED", "STATUS_AUTO", "STATUS_RAW",
]
