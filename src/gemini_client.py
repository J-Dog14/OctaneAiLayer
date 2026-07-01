"""
Thin wrapper around the supported `google-genai` SDK that logs every call to
ai_layer.llm_call_log and enforces a daily cost cap.

Public interface (unchanged from the old version, so callers don't care):
    generate(system_prompt, user_content, *, model=None,
             generated_report_id=None, response_mime_type=None) -> dict
    embed(text, *, model=None) -> list[float]

References:
- https://ai.google.dev/gemini-api/docs/migrate
- https://googleapis.github.io/python-genai/
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.db import backend_conn, exec_sql, query

load_dotenv()

# A single client instance is reused for the life of the process.
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Approximate per-million-token pricing (USD). Update if Google's pricing changes.
# Source: https://ai.google.dev/pricing
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash":      {"in": 0.075, "out": 0.30},
    "gemini-2.0-flash-lite": {"in": 0.075, "out": 0.30},
    "gemini-2.5-flash":      {"in": 0.30,  "out": 2.50},
    "gemini-2.5-pro":        {"in": 1.25,  "out": 10.0},
    "text-embedding-004":    {"in": 0.0,   "out": 0.0},
}


def _estimate_cost(model: str, in_toks: int, out_toks: int) -> float:
    p = PRICING.get(model, {"in": 0.0, "out": 0.0})
    return (in_toks * p["in"] + out_toks * p["out"]) / 1_000_000


def check_daily_cost_cap() -> None:
    """Raise if today's total spend has exceeded DAILY_COST_CAP_USD."""
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
    max_output_tokens: int | None = None,
    response_schema: Any = None,
) -> dict[str, Any]:
    """Run a single chat completion against Gemini and log the call.

    Returns a dict:
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

    cfg_kwargs: dict[str, Any] = {"system_instruction": system_prompt}
    if response_mime_type:
        cfg_kwargs["response_mime_type"] = response_mime_type
    if max_output_tokens is not None:
        cfg_kwargs["max_output_tokens"] = max_output_tokens
    if response_schema is not None:
        # Structured output — the model is constrained to produce JSON that
        # conforms to this schema. Dramatically reduces parse failures.
        # Accepts Pydantic model classes or dict schemas.
        cfg_kwargs["response_schema"] = response_schema
        # response_schema requires application/json mime
        if "response_mime_type" not in cfg_kwargs:
            cfg_kwargs["response_mime_type"] = "application/json"
    config = types.GenerateContentConfig(**cfg_kwargs)

    t0 = time.time()
    status = "success"
    err_msg: str | None = None
    text = ""
    in_toks = 0
    out_toks = 0
    raised: Exception | None = None

    # Retry on transient 503/UNAVAILABLE — Gemini's free-tier capacity spikes
    # are routine. Exponential backoff up to 6 attempts with ±20% jitter to
    # avoid thundering-herd retries all hitting Gemini in lockstep.
    # Base sequence: 2s, 4s, 8s, 16s, 32s (total ~62s of backoff)
    # With jitter: each sleep multiplied by [0.8 - 1.2]
    # If all 6 fail, the batch-level retry in eval_harness catches it 30s later.
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _client.models.generate_content(
                model=model,
                contents=user_content,
                config=config,
            )
            text = resp.text or ""
            usage = getattr(resp, "usage_metadata", None)
            if usage is not None:
                in_toks = getattr(usage, "prompt_token_count", 0) or 0
                out_toks = getattr(usage, "candidates_token_count", 0) or 0
            raised = None  # success
            break
        except Exception as e:
            err_str = str(e)
            is_transient = (
                "503" in err_str or "UNAVAILABLE" in err_str.upper()
                or "RESOURCE_EXHAUSTED" in err_str.upper() or "429" in err_str
            )
            if is_transient and attempt < max_attempts:
                # Exponential backoff (2s, 4s, 8s, 16s, 32s) with ±20% jitter.
                # Jitter spreads out concurrent retries so the entire process
                # isn't hammering Gemini at exactly the same moment when a
                # spike passes.
                base = 2 ** attempt  # 2, 4, 8, 16, 32 for attempts 1-5
                jitter_factor = random.uniform(0.8, 1.2)
                backoff = base * jitter_factor
                print(f"[gemini] transient error (attempt {attempt}/{max_attempts}), "
                      f"retrying in {backoff:.1f}s: {err_str[:120]}")
                time.sleep(backoff)
                continue
            status = "error"
            err_msg = err_str[:500]
            raised = e
            break

    latency_ms = int((time.time() - t0) * 1000)
    cost = _estimate_cost(model, in_toks, out_toks)

    # Always log, including failures
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
    resp = _client.models.embed_content(model=model, contents=text)
    # The new SDK returns ContentEmbedding objects in resp.embeddings; .values is the float list.
    return list(resp.embeddings[0].values)
