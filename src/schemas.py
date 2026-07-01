"""
Pydantic response schemas for structured Gemini output.

Passed to `gemini_client.generate(response_schema=...)` which constrains the
model to produce JSON conforming to these shapes. Dramatically reduces JSON
parse failures compared to just `response_mime_type="application/json"` alone.

Reasoning fields are annotated to hint short output — the LLM still needs the
prompt to enforce brevity but the schema advertises the intent.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────
# Common exercise shape (used by prep / BP / hitting / ME)
# ──────────────────────────────────────────────────────────────────────────

class ExercisePick(BaseModel):
    """One exercise selection with dose + slot + rationale."""
    exercise_name: str = Field(
        description="Exact exercise name from the candidate pool (case must match)"
    )
    slot: str = Field(
        default="",
        description="Slot label per the skill (e.g. 'mobility', 'activation', 'cuff')"
    )
    sets: int | None = Field(
        default=None, description="Set count, or null for time-based"
    )
    reps: int | None = Field(
        default=None, description="Rep count, or null for time-based"
    )
    duration_seconds: int | None = Field(
        default=None, description="Duration in seconds, or null for rep-based"
    )
    rationale: str = Field(
        default="",
        description="1-2 sentences tying this pick to the athlete's specific deficit"
    )


class ExerciseComponentOutput(BaseModel):
    """Standard output shape for prep / BP / hitting / ME generators."""
    reasoning: str = Field(
        description="1-3 sentences summarizing the athlete's dominant deficits "
                    "and how the exercise selection addresses them"
    )
    exercises: list[ExercisePick] = Field(
        description="List of exercise picks, in intended session order"
    )


# ──────────────────────────────────────────────────────────────────────────
# Plyo — nested structure (3 sessions + weekly layout)
# ──────────────────────────────────────────────────────────────────────────

class PlyoDrill(BaseModel):
    exercise_name: str = Field(description="Exact drill name from the pool")
    ball_weight: str = Field(
        default="",
        description="Ball weight display string (e.g. '5 oz', '32 oz')"
    )
    sets: int = Field(default=1, description="Set count (typically 1-2)")
    reps: int = Field(default=6, description="Rep count (typically 6-10)")
    order: int = Field(default=0, description="Order within the session (0-indexed)")
    rationale: str = Field(
        default="",
        description="1-2 sentences tying this drill to the athlete's inefficiency / feel goal"
    )


class PlyoSession(BaseModel):
    plyo_level: Literal["P0", "P1", "P2", "P3"]
    label: str = Field(default="", description="Short label (e.g. 'Recovery day')")
    session_intent: str = Field(
        default="",
        description="1-2 sentences summarizing the day's goal — what we're correcting, "
                    "what we want the athlete to feel"
    )
    drills: list[PlyoDrill]


class WeeklyLayout(BaseModel):
    """One-line plyo level assignment per day of week."""
    MONDAY: str = Field(default="", description="e.g. 'P0'")
    TUESDAY: str = Field(default="")
    WEDNESDAY: str = Field(default="")
    THURSDAY: str = Field(default="")
    FRIDAY: str = Field(default="")
    SATURDAY: str = Field(default="")
    SUNDAY: str = Field(default="")


class PlyoOutput(BaseModel):
    reasoning: str = Field(
        description="2-4 sentences. Identify the dominant deficit targeted, the feel "
                    "goal, any phase-based downgrades, and how the cycle addresses the "
                    "athlete's needs across the three intensities."
    )
    annual_phase_used: str = Field(default="")
    cycle: list[PlyoSession]
    weekly_layout: WeeklyLayout


# ──────────────────────────────────────────────────────────────────────────
# Lift template selection
# ──────────────────────────────────────────────────────────────────────────

class LiftTemplateSelection(BaseModel):
    template_id: str = Field(description="e.g. '5227-11'")
    movement_bucket: str = Field(
        description="Legs / Upper / Total Body / Sprint / Jump"
    )
    level: str = Field(default="", description="e.g. 'L1'")
    rationale: str = Field(
        default="",
        description="1-2 sentences explaining why this template for this athlete"
    )


class LiftTemplateOutput(BaseModel):
    reasoning: str = Field(
        description="1-3 sentences explaining the template family chosen and why "
                    "this athlete's profile suggested it"
    )
    primary_family_id: str = Field(
        default="",
        description="Family id chosen (e.g. '5227')"
    )
    primary_family_rationale: str = Field(
        default="",
        description="1-2 sentences explaining the family choice"
    )
    selected_templates: list[LiftTemplateSelection] = Field(
        description="5 templates — one per movement bucket (Legs/Upper/Total/Sprint/Jump)"
    )
