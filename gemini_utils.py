"""
Shared Google Gemini text generation helpers.

Uses the Generative Language REST API directly via httpx — no `google-generativeai`
SDK. This keeps the Vercel serverless bundle small (the SDK pulls in gRPC +
protobuf + google-auth, ~150 MB) and avoids the SDK's deprecation warning.
"""

import os
import re
import time

import httpx

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash-lite"
# Lite first — separate free-tier quota bucket; often less congested than flash.
GEMINI_MODEL_FALLBACKS = (
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
)


def _is_not_found_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "404" in err
        or "not found" in err
        or "notfound" in type(exc).__name__.lower()
    )


def _is_quota_error(exc: Exception) -> bool:
    err = str(exc).lower()
    return (
        "429" in err
        or "quota" in err
        or "rate limit" in err
        or "resourceexhausted" in type(exc).__name__.lower()
    )


def _retry_delay_seconds(exc: Exception) -> float:
    text = str(exc)
    for pattern in (
        r"retry in ([\d.]+)s",
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        r"seconds:\s*(\d+)\s*\]",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(float(match.group(1)), 60.0)
    return 5.0


def format_gemini_error(exc: Exception) -> str:
    """User-facing explanation for common Gemini API failures."""
    if _is_quota_error(exc):
        return (
            "**Gemini API quota exceeded** (free-tier rate or daily limits).\n\n"
            "This is not a bug in PitchCoach — your Google API key has no remaining "
            "free quota for the model that was tried.\n\n"
            "**Try:**\n"
            "1. Wait a few minutes and retry (minute limits reset quickly).\n"
            "2. Try a lighter model with a separate quota bucket:\n"
            "   `export GEMINI_MODEL=gemini-2.0-flash-lite`\n"
            "3. Check usage and limits: https://ai.dev/rate-limit\n"
            "4. Enable billing in [Google AI Studio](https://aistudio.google.com/) "
            "if you need higher limits.\n\n"
            f"_Details: {exc}_"
        )

    if _is_not_found_error(exc):
        return (
            "**Gemini model not found** for your API key.\n\n"
            "Set a current model name, e.g.:\n"
            "`export GEMINI_MODEL=gemini-2.0-flash-lite`\n\n"
            f"_Details: {exc}_"
        )

    return f"Gemini feedback failed: {exc}"


def _finish_reason_name(data: dict) -> str:
    """Extract the first candidate's finishReason from a REST JSON response."""
    try:
        return str(data["candidates"][0].get("finishReason", "")).upper()
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_text(data: dict) -> str:
    """
    Pull concatenated text out of a Gemini REST JSON response.

    Accumulates all text parts so a partial answer is never lost and an empty
    answer is detectable.
    """
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    chunks = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "".join(c for c in chunks if c).strip()


def generate_gemini_text(api_key: str, prompt: str, max_output_tokens: int = 2048) -> str:
    """
    Call the Gemini REST API with model fallbacks and brief retries on rate limits.

    Robust against the "thinking budget" pitfall: gemini-2.5 models spend output
    tokens on internal reasoning, so a small max_output_tokens can be exhausted
    before any answer is emitted (producing a truncated "1." style fragment).
    We start with a generous budget and, if a response is cut off by MAX_TOKENS
    with little/no usable text, retry once with a doubled budget.
    """
    preferred = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    candidates = [preferred]
    for name in GEMINI_MODEL_FALLBACKS:
        if name not in candidates:
            candidates.append(name)

    last_error: Exception | None = None
    for model_name in candidates:
        token_budget = max_output_tokens
        for attempt in range(2):
            try:
                url = f"{GEMINI_API_BASE}/{model_name}:generateContent"
                resp = httpx.post(
                    url,
                    headers={"x-goog-api-key": api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"maxOutputTokens": token_budget},
                    },
                    timeout=60,
                )
                # Raise for HTTP errors so the quota/404 handlers below can classify
                # them (the error string carries the status code and message).
                resp.raise_for_status()
                data = resp.json()

                text = _extract_text(data)
                finish = _finish_reason_name(data)

                # Truncated before producing a complete answer: the token budget
                # was eaten (commonly by 2.5 thinking). Retry once with more room.
                if finish == "MAX_TOKENS" and len(text) < 40:
                    if attempt == 0:
                        token_budget *= 2
                        continue
                    last_error = RuntimeError(
                        f"{model_name} hit MAX_TOKENS before producing an answer "
                        f"(reasoning consumed the {token_budget}-token budget)."
                    )
                    break

                if text:
                    return text

                last_error = RuntimeError(
                    f"{model_name} returned no text (finish_reason={finish or 'unknown'})."
                )
                break
            except httpx.HTTPStatusError as exc:
                # Include the response body so _is_quota_error / _is_not_found_error
                # can classify it (status code alone is in the message too).
                last_error = RuntimeError(
                    f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                )
                if _is_not_found_error(last_error):
                    break  # try next model
                if _is_quota_error(last_error) and attempt == 0:
                    time.sleep(_retry_delay_seconds(last_error))
                    continue
                if _is_quota_error(last_error):
                    break
                raise last_error from exc
            except httpx.HTTPError as exc:
                # Network/timeout — retry once, then give up on this model.
                last_error = exc
                if attempt == 0:
                    time.sleep(2)
                    continue
                break

    if last_error is not None:
        raise last_error
    raise RuntimeError("No Gemini model candidates were available.")
