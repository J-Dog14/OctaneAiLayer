"""
Thin wrapper around the Gemini SDK that logs every call to ai_layer.llm_call_log
and enforces a daily cost cap.

Use:
    from src.gemini_client import generate, embed
    result = generate(system_prompt, user_content, generated_report_id=42)
    vec = embed("some text to embed")
"""
from __future__ import annotations

import os
import time
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from src.db import backend_conn, exec_sql, query

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Approximate per-million-token pricing (USD). Update if Google's pricing changes.
# Source: https://ai.google.dev/pricing — verify before treating these as exact.
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash":       {"in": 0.075, "out": 0.30},
    "gemini-2.0-flash-lite":  {"in": 0.075, "out": 0.30},
    "gemini-2.5-flash":       {"in": 0.30,  "out": 2.50},
    "gemini-2.5-pro":         {"in": 1.25,  "out": 10.0},
    "text-embedding-004":     {"in": 0.0,   "out": 0.0},   # free tier as of writing
}


def _estimate_cost(model: str, in_toks: int, out_toks: int) -> float:
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    return (in_toks * p["in"] + out_toks * p["out"]) / 1_000_000


def check_daily_cost_cap() -> None:
    """Raise if today's total spend exceeds DAILY_COST_CAP_USD."""
    cap = float(os.getenv("DAILY_COST_CAP_USD", "5.00"))
    with backend_conn() as conn:
        rows = query(conn, """
            SELECT COALESCE(SUM(cost_usd), 0)::float AS spent
            FROM ai_layer.llm_call_log
            WHERE created_at > now() - interval '1 day'
        """)
    spent = float(rows[0]["spent"])
    if spent >= cap:
        raise RuntimeError(
            f"Daily cost cap reached: ${spent:.4f} >= ${cap:.2f}. "
            f"Raise DAILY_COST_CAP_USD in .env or wait until tomorrow."
        )


def generate(
    system_prompt: str,
    user_content: str,
    *,
    model: str | None = None,
    generated_report_id: int | None = None,
    response_mime_type: str | None = None,
) -> dict[str, Any]:
    """Run a single chat completion against Gemini and log the call.

    Returns:
        {
          "text": str,
          "input_tokens": int,
          "output_tokens": int,
          "latency_ms": int,
          "cost_usd": float,
          "model": str,
        }
    """
    check_daily_cost_cap()

    model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gen_config: dict[str, Any] = {}
    if response_mime_type:
        gen_config["response_mime_type"] = response_mime_type

    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,
        generation_config=gen_config or None,
    )

    t0 = time.time()
    status = "success"
    err_msg: str | None = None
    text = ""
    in_toks = 0
    out_toks = 0
    raised: Exception | None = None

    try:
        resp = gm.generate_content(user_content)
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        if usage is not None:
            in_toks = getattr(usage, "prompt_token_count", 0) or 0
            out_toks = getattr(usage, "candidates_token_count", 0) or 0
    except Exception as e:
        status = "error"
        err_msg = str(e)[:500]
        raised = e

    latency_ms = int((time.time() - t0) * 1000)
    cost = _estimate_cost(model, in_toks, out_toks)

    # Always log, even on error
    with backend_conn() as conn:
        exec_sql(conn, """
            INSERT INTO ai_layer.llm_call_log
              (generated_report_id, model_name, input_tokens, output_tokens,
               cost_usd, latency_ms, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, [generated_report_id, model, in_toks, out_toks,
              cost, latency_ms, status, err_msg])

    if raised:
        raise raised

    return {
        "text": text,
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "model": model,
    }


def embed(text: str, *, model: str | None = None) -> list[float]:
    """Return the embedding vector for `text` (768 dims for text-embedding-004)."""
    model = model or os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
    resp = genai.embed_content(model=f"models/{model}", content=text)
    return resp["embedding"]
