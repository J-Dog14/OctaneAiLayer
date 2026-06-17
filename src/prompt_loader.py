"""
Load a skill from skills/<name>/ and register it in ai_layer.prompt_versions.

Each skill folder may contain:
  - SKILL.md                            (preferred, raw markdown)
  - <name>.skill                        (zip archive; we extract SKILL.md from it)
  - references/*.md                     (appended in sorted order)
  - references/*.json                   (appended as fenced JSON blocks)

The full concatenated text is hashed; if the hash matches the latest stored
version, we reuse it. Otherwise we insert a new version row and return that id.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from src.db import backend_conn, query, returning_id

SKILLS_ROOT = Path(__file__).resolve().parents[1] / "skills"


def _read_main_skill_md(skill_dir: Path, skill_name: str) -> str:
    """Return the SKILL.md contents, extracting from a .skill zip if needed."""
    plain = skill_dir / "SKILL.md"
    if plain.exists():
        return plain.read_text(encoding="utf-8")

    zip_candidate = skill_dir / f"{skill_name}.skill"
    if zip_candidate.exists():
        with zipfile.ZipFile(zip_candidate) as zf:
            # Try both <skill>/SKILL.md and SKILL.md inside the zip
            candidates = [f"{skill_name}/SKILL.md", "SKILL.md"]
            for c in candidates:
                if c in zf.namelist():
                    return zf.read(c).decode("utf-8")
            # Fallback: any path ending in SKILL.md
            for member in zf.namelist():
                if member.endswith("SKILL.md"):
                    return zf.read(member).decode("utf-8")

    raise FileNotFoundError(
        f"No SKILL.md found for '{skill_name}' (looked in {plain} and {zip_candidate})"
    )


def _read_skill_text(skill_name: str) -> str:
    """Combine SKILL.md + every references/*.md and references/*.json into one prompt."""
    skill_dir = SKILLS_ROOT / skill_name
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

    parts: list[str] = []
    parts.append(f"# Skill: {skill_name}\n\n")
    parts.append(_read_main_skill_md(skill_dir, skill_name))

    refs_dir = skill_dir / "references"
    if refs_dir.exists():
        for ref in sorted(refs_dir.glob("*.md")):
            parts.append(f"\n\n## Reference: {ref.stem}\n\n")
            parts.append(ref.read_text(encoding="utf-8"))
        for ref in sorted(refs_dir.glob("*.json")):
            parts.append(f"\n\n## Reference data: {ref.stem}\n\n```json\n")
            parts.append(ref.read_text(encoding="utf-8"))
            parts.append("\n```\n")

    return "".join(parts)


def load_and_register(skill_name: str) -> tuple[str, int]:
    """Load the skill prompt from disk and ensure it's recorded in the DB.

    Returns (prompt_text, prompt_version_id).
    """
    text = _read_skill_text(skill_name)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()

    with backend_conn() as conn:
        rows = query(conn, """
            SELECT id, version, prompt_hash
            FROM ai_layer.prompt_versions
            WHERE skill_name = %s
            ORDER BY version DESC
            LIMIT 1
        """, [skill_name])

        if rows and rows[0]["prompt_hash"] == h:
            return text, rows[0]["id"]

        next_version = (rows[0]["version"] + 1) if rows else 1
        new_id = returning_id(conn, """
            INSERT INTO ai_layer.prompt_versions
              (skill_name, version, prompt_text, prompt_hash, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, [skill_name, next_version, text, h,
              f"Auto-registered v{next_version} from skills/{skill_name}/"])

    return text, new_id
