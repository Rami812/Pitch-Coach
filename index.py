"""
PitchCoach — SELF-CONTAINED FastAPI function for Vercel.

Everything the function needs (frontend HTML, the XGBoost model, feature
engineering, and Gemini calls) is inlined into THIS ONE FILE. There are no
imports of local modules and no includeFiles dependency, so nothing can be
left out of the serverless bundle.

Remote services (set as env vars in the Vercel dashboard):
  - Audio transcription → Groq Whisper Large V3          (GROQ_API_KEY)
  - BERT embeddings     → HuggingFace feature-extraction  (HF_API_KEY + HF_MODEL_ID)
  - Coaching feedback   → Google Gemini API               (GOOGLE_API_KEY)
"""

import io
import math
import os
import re
import time

import httpx
import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# ── inlined feature engineering (from feature_utils.py) ──────────────────────
import re

EMBEDDING_DIM = 768  # DistilBERT hidden size

FILLER_WORDS = {
    "um", "uh", "like", "basically", "literally", "actually", "honestly",
    "you know", "i mean", "sort of", "kind of", "right", "okay", "so",
    "anyway", "whatever", "stuff", "things", "very", "really", "just",
}

HEDGE_PHRASES = [
    "i think", "i guess", "i believe", "maybe", "perhaps", "possibly",
    "might", "could be", "not sure", "i hope", "trying to", "kind of",
    "sort of", "a bit", "somewhat",
]

# Fixed ordering of the numeric linguistic features handed to XGBoost.
# DO NOT reorder — the trained model expects this exact sequence.
LINGUISTIC_FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "filler_word_count",
    "filler_ratio",
    "hedge_count",
    "data_point_count",
    "question_count",
]


def clean_text(text: str) -> str:
    text = str(text).replace("“", "").replace("”", "")  # strip smart quotes
    return re.sub(r"\s+", " ", text).strip()


def extract_text_features(text: str) -> dict:
    """Extract human-readable linguistic signals that correlate with pitch quality."""
    lower = text.lower()
    words = lower.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    found_fillers = [w for w in words if w in FILLER_WORDS]
    found_hedges = [p for p in HEDGE_PHRASES if p in lower]
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?(?:\s?%|x|k|m|b)?\b", lower)
    question_count = text.count("?")
    avg_sentence_len = (len(words) / len(sentences)) if sentences else 0

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "filler_words_found": sorted(set(found_fillers)),
        "filler_word_count": len(found_fillers),
        "hedge_phrases_found": sorted(set(found_hedges)),
        "data_points": list(dict.fromkeys(numbers))[:10],  # unique, cap at 10
        "question_count": question_count,
    }


def features_to_vector(feat: dict) -> list:
    """Convert a feature dict into the fixed-order numeric vector for XGBoost."""
    word_count = feat.get("word_count", 0) or 0
    filler_count = feat.get("filler_word_count", 0) or 0
    return [
        float(word_count),
        float(feat.get("sentence_count", 0) or 0),
        float(feat.get("avg_sentence_length", 0) or 0),
        float(filler_count),
        float(filler_count / word_count) if word_count else 0.0,  # filler_ratio
        float(len(feat.get("hedge_phrases_found", []))),           # hedge_count
        float(len(feat.get("data_points", []))),                   # data_point_count
        float(feat.get("question_count", 0) or 0),
    ]


def positive_prob(score_output) -> float:
    """
    Normalize the pure-Python evaluator's `score()` result into P(good).

    score() returns a scalar probability of the positive (good) class; guard
    against a list form too, and clamp to [0, 1].
    """
    if isinstance(score_output, (list, tuple)):
        p = score_output[-1] if len(score_output) >= 2 else score_output[0]
    else:
        p = score_output
    return min(1.0, max(0.0, float(p)))


def pool_hf_embedding(hf_response) -> list:
    """
    Mean-pool a HuggingFace feature-extraction response into one 768-d vector.

    The pipeline returns token-level embeddings whose shape can be (tokens, 768),
    (1, tokens, 768), or already-pooled (768,). Pure Python (no numpy) so the
    Vercel serverless function stays dependency-light.
    """
    def depth(x) -> int:
        d = 0
        while isinstance(x, list):
            if not x:
                break
            x = x[0]
            d += 1
        return d

    arr = hf_response
    while depth(arr) > 2:          # e.g. (1, tokens, 768) -> (tokens, 768)
        arr = arr[0]

    if depth(arr) == 2:            # (tokens, 768) -> mean over tokens -> (768,)
        n = len(arr)
        pooled = [0.0] * len(arr[0])
        for row in arr:
            for i, v in enumerate(row):
                pooled[i] += v
        pooled = [v / n for v in pooled]
    else:                          # already 1-D (768,)
        pooled = list(arr)

    if len(pooled) != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding size {len(pooled)}, expected {EMBEDDING_DIM}."
        )
    return [float(v) for v in pooled]

# ── inlined Gemini client (from gemini_utils.py) ─────────────────────────────
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

# ── embedded XGBoost model (pure-Python tree evaluator) ──────────────────────
import json as _json
_MODEL = _json.loads('{"base_margin": -0.07232062490520737, "trees": [{"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0482854061}, "2": {"leaf": -0.0451312028}}, {"0": {"f": 223, "thr": 0.305217028, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0433605611}, "2": {"leaf": -0.0441186726}}, {"0": {"f": 592, "thr": 0.00659042597, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0419416465}, "2": {"leaf": -0.0423143804}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0458154529}, "2": {"leaf": -0.0441186689}}, {"0": {"f": 460, "thr": -0.134202793, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0426909737}, "2": {"leaf": 0.0421140529}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0443257764}, "2": {"leaf": -0.0425778553}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0423663706}, "2": {"leaf": -0.0401345529}}, {"0": {"f": 65, "thr": 0.123210095, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.039601054}, "2": {"leaf": 0.0376173072}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0416325964}, "2": {"leaf": -0.0378322452}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0405541994}, "2": {"leaf": -0.0386454687}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0401909761}, "2": {"leaf": -0.038788151}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0372673795}, "2": {"leaf": -0.0376661569}}, {"0": {"f": 371, "thr": 0.10966973, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0364618823}, "2": {"leaf": 0.0345858708}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.036900539}, "2": {"leaf": -0.0360012427}}, {"0": {"f": 65, "thr": 0.123210095, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0336742066}, "2": {"leaf": 0.0371923968}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0361203104}, "2": {"leaf": -0.0356116854}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0346399248}, "2": {"leaf": -0.034645848}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0346693173}, "2": {"leaf": -0.0338738225}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0344518609}, "2": {"leaf": -0.0325096548}}, {"0": {"f": 65, "thr": 0.185413644, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.032992024}, "2": {"leaf": 0.0325977802}}, {"0": {"f": 75, "thr": -0.130147979, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0331564024}, "2": {"leaf": -0.0312977433}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0325954631}, "2": {"leaf": -0.0323333144}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0335417166}, "2": {"leaf": -0.0311259292}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0321945287}, "2": {"leaf": -0.0318935402}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0312203933}, "2": {"leaf": -0.0305856373}}, {"0": {"f": 223, "thr": 0.305217028, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0290149115}, "2": {"leaf": -0.0306267031}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0301088896}, "2": {"leaf": -0.0303213317}}, {"0": {"f": 214, "thr": -0.161071375, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.031398017}, "2": {"leaf": -0.0294221286}}, {"0": {"f": 460, "thr": -0.121036254, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0290295612}, "2": {"leaf": 0.0273519997}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0306324996}, "2": {"leaf": -0.0293647405}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0292135086}, "2": {"leaf": -0.0286553614}}, {"0": {"f": 460, "thr": -0.121036254, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0285211504}, "2": {"leaf": 0.0261797383}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0289386474}, "2": {"leaf": -0.0281364843}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0289283171}, "2": {"leaf": -0.0282566622}}, {"0": {"f": 65, "thr": 0.123210095, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0247805342}, "2": {"leaf": 0.0277058855}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0274993721}, "2": {"leaf": -0.0269204639}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0278130416}, "2": {"leaf": -0.0254870541}}, {"0": {"f": 371, "thr": 0.10966973, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0269582681}, "2": {"leaf": 0.026774805}}, {"0": {"f": 425, "thr": 0.509111941, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0249091852}, "2": {"leaf": 0.0247545727}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0259419009}, "2": {"leaf": -0.0263179671}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0263110138}, "2": {"leaf": -0.0250683799}}, {"0": {"f": 214, "thr": -0.110274702, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0236308761}, "2": {"leaf": -0.025295008}}, {"0": {"f": 583, "thr": 0.219173506, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0236193407}, "2": {"leaf": 0.025599597}}, {"0": {"f": 561, "thr": -0.272860736, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0234507248}, "2": {"leaf": -0.0248054322}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"f": 65, "thr": 0.123210095, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0228280202}, "2": {"leaf": 0.0251480788}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0253036451}, "2": {"leaf": -0.0243749209}}, {"0": {"leaf": 0.00111543434}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0250856783}, "2": {"leaf": -0.0244451519}}, {"0": {"f": 552, "thr": 0.185839653, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0243487731}, "2": {"leaf": 0.0227576848}}, {"0": {"f": 214, "thr": -0.122106463, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0251132194}, "2": {"leaf": -0.0246831849}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0244274791}, "2": {"leaf": -0.0238431245}}, {"0": {"leaf": 0.00257279584}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00283391331}}, {"0": {"leaf": 0.00143650221}}, {"0": {"leaf": 0}}, {"0": {"f": 561, "thr": -0.272860736, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0226797909}, "2": {"leaf": -0.0215522479}}, {"0": {"leaf": -0.000692697184}}, {"0": {"leaf": 0.000441807846}}, {"0": {"leaf": 0.000243182032}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00105746533}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0242681727}, "2": {"leaf": -0.0239433739}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000190243096}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000627198548}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00026090446}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00144563254}}, {"0": {"leaf": 0.00233251997}}, {"0": {"leaf": -8.77667699e-05}}, {"0": {"f": 94, "thr": 0.123538919, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0208182614}, "2": {"leaf": -0.0214188937}}, {"0": {"leaf": 0}}, {"0": {"leaf": -7.87507161e-05}}, {"0": {"leaf": -0.000123772086}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000396426563}}, {"0": {"leaf": -0.000632510521}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000168432278}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000501420815}}, {"0": {"leaf": -0.00385011872}}, {"0": {"leaf": -0.00108457275}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000377464865}}, {"0": {"leaf": 0.000260347617}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00057473348}}, {"0": {"leaf": 0.000102204162}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000594831305}}, {"0": {"f": 592, "thr": -0.0383866429, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0239207055}, "2": {"leaf": -0.019157974}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00026618471}}, {"0": {"leaf": -0.00021925148}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -4.16295843e-05}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000567042385}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000810234749}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000835263811}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000364704756}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -8.27188051e-05}}, {"0": {"leaf": 0.000868401432}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00108657626}}, {"0": {"leaf": 6.85597479e-05}}, {"0": {"leaf": -0.000145227736}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.0021948549}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00106345373}}, {"0": {"leaf": -0.00049502094}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000244540308}}, {"0": {"leaf": 0.000504660129}}, {"0": {"leaf": 0.000546345778}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00102104305}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.0008339372}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.00143696903}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000910785282}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.000236302527}}, {"0": {"leaf": 9.20858147e-05}}, {"0": {"leaf": 0.00212998525}}, {"0": {"leaf": 0}}, {"0": {"f": 214, "thr": -0.117862873, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": 0.0241379067}, "2": {"leaf": -0.0236985218}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00049178832}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.000163933801}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0}}, {"0": {"leaf": -0.0012613961}}, {"0": {"leaf": 0}}, {"0": {"leaf": 0.00375241041}}, {"0": {"f": 371, "thr": 0.10966973, "yes": 1, "no": 2, "missing": 2}, "1": {"leaf": -0.0216212012}, "2": {"leaf": 0.0219530743}}]}')
_BASE_MARGIN = _MODEL["base_margin"]
_TREES = _MODEL["trees"]

def _leaf_value(tree, features):
    node = tree["0"]
    while "leaf" not in node:
        v = features[node["f"]]
        if v is None:
            nxt = node["missing"]
        elif v < node["thr"]:
            nxt = node["yes"]
        else:
            nxt = node["no"]
        node = tree[str(nxt)]
    return node["leaf"]

def score(features):
    margin = _BASE_MARGIN + sum(_leaf_value(t, features) for t in _TREES)
    return 1.0 / (1.0 + math.exp(-margin))


# ── embedded frontend ─────────────────────────────────────────────────────────
_INDEX_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n  <meta charset="UTF-8" />\n  <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n  <title>PitchCoach</title>\n  <style>\n    :root {\n      --bg: #0f1117;\n      --surface: #1e2130;\n      --surface2: #262a3a;\n      --border: #2e3248;\n      --accent: #7c6af7;\n      --accent-dim: #4f46a8;\n      --good: #22c55e;\n      --bad: #ef4444;\n      --text: #e8eaf0;\n      --text-dim: #8892a4;\n      --radius: 12px;\n    }\n\n    * { box-sizing: border-box; margin: 0; padding: 0; }\n\n    body {\n      background: var(--bg);\n      color: var(--text);\n      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;\n      min-height: 100vh;\n      display: flex;\n      flex-direction: column;\n      align-items: center;\n      padding: 2rem 1rem;\n    }\n\n    header {\n      text-align: center;\n      margin-bottom: 2.5rem;\n    }\n    header h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }\n    header h1 span { color: var(--accent); }\n    header p  { color: var(--text-dim); margin-top: 0.5rem; font-size: 1rem; max-width: 500px; }\n\n    .card {\n      background: var(--surface);\n      border: 1px solid var(--border);\n      border-radius: var(--radius);\n      padding: 1.5rem;\n      width: 100%;\n      max-width: 680px;\n      margin-bottom: 1.25rem;\n    }\n\n    .card h2 {\n      font-size: 0.8rem;\n      text-transform: uppercase;\n      letter-spacing: 1px;\n      color: var(--text-dim);\n      margin-bottom: 1rem;\n    }\n\n    /* ── upload zone ─────────────────────────────────────────────── */\n    .drop-zone {\n      border: 2px dashed var(--border);\n      border-radius: var(--radius);\n      padding: 2.5rem 1rem;\n      text-align: center;\n      cursor: pointer;\n      transition: border-color 0.2s, background 0.2s;\n    }\n    .drop-zone:hover, .drop-zone.dragover {\n      border-color: var(--accent);\n      background: rgba(124, 106, 247, 0.05);\n    }\n    .drop-zone .icon { font-size: 2.5rem; display: block; margin-bottom: 0.75rem; }\n    .drop-zone p { color: var(--text-dim); font-size: 0.9rem; }\n    .drop-zone p strong { color: var(--text); }\n    #file-input { display: none; }\n\n    .file-info {\n      display: none;\n      margin-top: 1rem;\n      padding: 0.75rem 1rem;\n      background: var(--surface2);\n      border-radius: 8px;\n      font-size: 0.875rem;\n      align-items: center;\n      gap: 0.75rem;\n    }\n    .file-info.show { display: flex; }\n    .file-info .fname { font-weight: 500; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }\n    .file-info .fsize { color: var(--text-dim); white-space: nowrap; }\n\n    audio { width: 100%; margin-top: 1rem; border-radius: 8px; display: none; }\n    audio.show { display: block; }\n\n    /* ── button ──────────────────────────────────────────────────── */\n    button.primary {\n      width: 100%;\n      max-width: 680px;\n      padding: 0.85rem;\n      background: var(--accent);\n      color: #fff;\n      border: none;\n      border-radius: var(--radius);\n      font-size: 1rem;\n      font-weight: 600;\n      cursor: pointer;\n      transition: background 0.2s, opacity 0.2s;\n      margin-bottom: 1.25rem;\n    }\n    button.primary:hover:not(:disabled) { background: var(--accent-dim); }\n    button.primary:disabled { opacity: 0.45; cursor: not-allowed; }\n\n    /* ── spinner ─────────────────────────────────────────────────── */\n    .spinner {\n      display: none;\n      align-items: center;\n      gap: 0.75rem;\n      color: var(--text-dim);\n      font-size: 0.9rem;\n      justify-content: center;\n      margin-bottom: 1.25rem;\n    }\n    .spinner.show { display: flex; }\n    .spinner-ring {\n      width: 20px; height: 20px;\n      border: 2px solid var(--border);\n      border-top-color: var(--accent);\n      border-radius: 50%;\n      animation: spin 0.7s linear infinite;\n    }\n    @keyframes spin { to { transform: rotate(360deg); } }\n\n    /* ── error banner ────────────────────────────────────────────── */\n    .error-banner {\n      display: none;\n      width: 100%; max-width: 680px;\n      background: rgba(239, 68, 68, 0.12);\n      border: 1px solid var(--bad);\n      border-radius: var(--radius);\n      padding: 0.85rem 1rem;\n      color: var(--bad);\n      font-size: 0.9rem;\n      margin-bottom: 1.25rem;\n    }\n    .error-banner.show { display: block; }\n\n    /* ── results ─────────────────────────────────────────────────── */\n    #results { display: none; width: 100%; max-width: 680px; }\n    #results.show { display: block; }\n\n    .transcript-text {\n      background: var(--surface2);\n      border-radius: 8px;\n      padding: 1rem;\n      font-size: 0.9rem;\n      line-height: 1.6;\n      color: var(--text);\n      max-height: 200px;\n      overflow-y: auto;\n    }\n    .word-count { color: var(--text-dim); font-size: 0.8rem; margin-top: 0.5rem; }\n\n    .verdict {\n      display: flex;\n      align-items: center;\n      gap: 1rem;\n      flex-wrap: wrap;\n    }\n    .badge {\n      display: inline-flex;\n      align-items: center;\n      gap: 0.4rem;\n      font-weight: 700;\n      font-size: 1rem;\n      padding: 0.5rem 1rem;\n      border-radius: 99px;\n    }\n    .badge.good { background: rgba(34, 197, 94, 0.15); color: var(--good); }\n    .badge.bad  { background: rgba(239, 68, 68, 0.15); color: var(--bad); }\n\n    .metrics { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.75rem; }\n    .metric { flex: 1; min-width: 130px; }\n    .metric .label { font-size: 0.75rem; color: var(--text-dim); margin-bottom: 0.3rem; }\n    .metric .value { font-size: 1.5rem; font-weight: 700; }\n    .metric .value.good { color: var(--good); }\n    .metric .value.bad  { color: var(--bad); }\n\n    .progress-bar-wrap {\n      background: var(--surface2);\n      border-radius: 99px;\n      height: 10px;\n      margin-top: 1rem;\n      overflow: hidden;\n    }\n    .progress-bar {\n      height: 100%;\n      border-radius: 99px;\n      background: linear-gradient(90deg, var(--accent-dim), var(--accent));\n      transition: width 0.6s ease;\n    }\n    .progress-label {\n      display: flex;\n      justify-content: space-between;\n      font-size: 0.75rem;\n      color: var(--text-dim);\n      margin-top: 0.35rem;\n    }\n\n    .interpretation {\n      font-size: 0.9rem;\n      line-height: 1.7;\n      color: var(--text-dim);\n    }\n    .interpretation strong { color: var(--text); }\n    .interpretation ul { margin-top: 0.5rem; padding-left: 1.2rem; }\n    .interpretation li { margin-bottom: 0.35rem; }\n\n    .feedback-text {\n      font-size: 0.9rem;\n      line-height: 1.75;\n      color: var(--text-dim);\n      white-space: pre-wrap;\n    }\n    .feedback-text strong { color: var(--text); }\n\n    footer {\n      margin-top: 3rem;\n      color: var(--text-dim);\n      font-size: 0.78rem;\n      text-align: center;\n    }\n  </style>\n</head>\n<body>\n\n<header>\n  <h1>🎙️ Pitch<span>Coach</span></h1>\n  <p>Upload a business pitch recording and get instant AI feedback powered by Whisper + BERT + Gemini.</p>\n</header>\n\n<!-- upload card -->\n<div class="card">\n  <h2>Your pitch recording</h2>\n  <div class="drop-zone" id="drop-zone" onclick="document.getElementById(\'file-input\').click()">\n    <span class="icon">🎤</span>\n    <p><strong>Click to upload</strong> or drag &amp; drop</p>\n    <p style="margin-top:0.3rem">WAV · MP3 · M4A · OGG · FLAC · WebM &nbsp;·&nbsp; Max ~4 MB</p>\n  </div>\n  <input type="file" id="file-input" accept=".wav,.mp3,.m4a,.ogg,.flac,.webm,audio/*" />\n  <div class="file-info" id="file-info">\n    <span>📄</span>\n    <span class="fname" id="file-name"></span>\n    <span class="fsize" id="file-size"></span>\n  </div>\n  <audio id="audio-player" controls></audio>\n</div>\n\n<button class="primary" id="analyze-btn" disabled>Analyze my pitch</button>\n\n<div class="spinner" id="spinner">\n  <div class="spinner-ring"></div>\n  <span id="spinner-msg">Uploading…</span>\n</div>\n\n<div class="error-banner" id="error-banner"></div>\n\n<!-- results -->\n<div id="results">\n  <div class="card">\n    <h2>Transcript</h2>\n    <div class="transcript-text" id="transcript"></div>\n    <p class="word-count" id="word-count"></p>\n  </div>\n\n  <div class="card">\n    <h2>Pitch quality</h2>\n    <div class="verdict">\n      <div class="badge" id="verdict-badge"></div>\n    </div>\n    <div class="metrics">\n      <div class="metric">\n        <div class="label">Good pitch score</div>\n        <div class="value good" id="good-pct"></div>\n      </div>\n      <div class="metric">\n        <div class="label">Needs-work score</div>\n        <div class="value bad" id="bad-pct"></div>\n      </div>\n    </div>\n    <div class="progress-bar-wrap">\n      <div class="progress-bar" id="progress-bar" style="width:0%"></div>\n    </div>\n    <div class="progress-label">\n      <span>Bad</span><span>Good</span>\n    </div>\n  </div>\n\n  <div class="card" id="interpretation-card">\n    <h2>What this means</h2>\n    <div class="interpretation" id="interpretation"></div>\n  </div>\n\n  <div class="card" id="feedback-card" style="display:none">\n    <h2>Gemini\'s coaching feedback</h2>\n    <div class="feedback-text" id="feedback"></div>\n  </div>\n</div>\n\n<footer>PitchCoach &nbsp;·&nbsp; DistilBERT fine-tuned on 83 entrepreneur pitches &nbsp;·&nbsp; Transcription by Groq Whisper Large V3 &nbsp;·&nbsp; Coaching by Gemini</footer>\n\n<script>\n  const dropZone   = document.getElementById("drop-zone");\n  const fileInput  = document.getElementById("file-input");\n  const fileInfo   = document.getElementById("file-info");\n  const fileName   = document.getElementById("file-name");\n  const fileSize   = document.getElementById("file-size");\n  const audioEl    = document.getElementById("audio-player");\n  const analyzeBtn = document.getElementById("analyze-btn");\n  const spinner    = document.getElementById("spinner");\n  const spinnerMsg = document.getElementById("spinner-msg");\n  const errorBanner= document.getElementById("error-banner");\n  const results    = document.getElementById("results");\n\n  let selectedFile = null;\n\n  function formatBytes(bytes) {\n    if (bytes < 1024) return bytes + " B";\n    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";\n    return (bytes / (1024 * 1024)).toFixed(1) + " MB";\n  }\n\n  function setFile(file) {\n    if (!file) return;\n    selectedFile = file;\n    fileName.textContent = file.name;\n    fileSize.textContent = formatBytes(file.size);\n    fileInfo.classList.add("show");\n    audioEl.src = URL.createObjectURL(file);\n    audioEl.classList.add("show");\n    analyzeBtn.disabled = false;\n    errorBanner.classList.remove("show");\n    results.classList.remove("show");\n  }\n\n  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));\n\n  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });\n  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));\n  dropZone.addEventListener("drop", (e) => {\n    e.preventDefault();\n    dropZone.classList.remove("dragover");\n    setFile(e.dataTransfer.files[0]);\n  });\n\n  function showError(msg) {\n    errorBanner.textContent = "⚠ " + msg;\n    errorBanner.classList.add("show");\n    spinner.classList.remove("show");\n    analyzeBtn.disabled = false;\n  }\n\n  function renderMarkdown(text) {\n    // minimal markdown: **bold**, bullet lists, newlines\n    return text\n      .replace(/\\*\\*(.*?)\\*\\*/g, "<strong>$1</strong>")\n      .replace(/^- (.+)$/gm, "<li>$1</li>")\n      .replace(/(<li>.*<\\/li>)/s, "<ul>$1</ul>")\n      .replace(/\\n/g, "<br>");\n  }\n\n  analyzeBtn.addEventListener("click", async () => {\n    if (!selectedFile) return;\n\n    const MAX_BYTES = 4.4 * 1024 * 1024;  // Vercel serverless body limit ~4.5 MB\n    if (selectedFile.size > MAX_BYTES) {\n      showError("Audio is " + formatBytes(selectedFile.size) + " — the server accepts up to ~4.4 MB. Upload a shorter or more compressed clip (MP3/M4A work best).");\n      return;\n    }\n\n    analyzeBtn.disabled = true;\n    errorBanner.classList.remove("show");\n    results.classList.remove("show");\n    spinner.classList.add("show");\n    spinnerMsg.textContent = "Transcribing with Whisper…";\n\n    const formData = new FormData();\n    formData.append("audio", selectedFile, selectedFile.name);\n\n    try {\n      spinnerMsg.textContent = "Classifying with BERT…";\n      const resp = await fetch("/api/analyze", { method: "POST", body: formData });\n      const raw = await resp.text();\n      let data;\n      try { data = JSON.parse(raw); }\n      catch (e) {\n        analyzeBtn.disabled = false;\n        spinner.classList.remove("show");\n        showError(resp.status === 413\n          ? "Audio too large for the server (Vercel ~4.5 MB limit). Use a shorter or compressed clip."\n          : ("Server error " + resp.status + ": " + raw.slice(0, 180)));\n        return;\n      }\n\n      if (!resp.ok) {\n        showError(data.detail || `Server error (${resp.status})`);\n        return;\n      }\n\n      spinner.classList.remove("show");\n      analyzeBtn.disabled = false;\n\n      // transcript\n      document.getElementById("transcript").textContent = data.transcript;\n      document.getElementById("word-count").textContent = `${data.word_count} words transcribed`;\n\n      // classification\n      const cls   = data.classification;\n      const label = cls.label;\n      const goodPct = Math.round(cls.good_prob * 100);\n      const badPct  = Math.round(cls.bad_prob  * 100);\n\n      const badge = document.getElementById("verdict-badge");\n      badge.textContent = label === "good" ? "✅ Good pitch" : "❌ Needs work";\n      badge.className = "badge " + label;\n\n      document.getElementById("good-pct").textContent = goodPct + "%";\n      document.getElementById("bad-pct").textContent  = badPct  + "%";\n      document.getElementById("progress-bar").style.width = goodPct + "%";\n\n      // interpretation\n      const interp = document.getElementById("interpretation");\n      if (label === "good") {\n        interp.innerHTML = `Your pitch scored <strong>${goodPct}% good</strong> — it shares characteristics with influential speeches in our training set.<br><br>\n          <strong>Common traits of good pitches:</strong>\n          <ul>\n            <li>Clear problem + solution framing</li>\n            <li>Confident, direct language with minimal hedging</li>\n            <li>Concrete specifics — numbers, names, timelines</li>\n            <li>A compelling call to action</li>\n          </ul>`;\n      } else {\n        interp.innerHTML = `Your pitch scored <strong>${badPct}% needs work</strong> — it shares traits with weaker pitches in our training set.<br><br>\n          <strong>Common weaknesses to address:</strong>\n          <ul>\n            <li>Vague language or excessive jargon</li>\n            <li>Lack of structure: problem → solution → traction → ask</li>\n            <li>Too many filler words or hedges ("sort of", "I think maybe")</li>\n            <li>Missing a clear call to action</li>\n          </ul>`;\n      }\n\n      // feedback\n      const feedbackCard = document.getElementById("feedback-card");\n      const feedbackEl   = document.getElementById("feedback");\n      if (data.feedback) {\n        feedbackEl.innerHTML = renderMarkdown(data.feedback);\n        feedbackCard.style.display = "block";\n      } else {\n        feedbackCard.style.display = "none";\n      }\n\n      results.classList.add("show");\n      results.scrollIntoView({ behavior: "smooth", block: "start" });\n\n    } catch (err) {\n      showError("Request failed: " + ((err && err.message) ? err.message : err) + " (it may have timed out).");\n      console.error(err);\n    }\n  });\n</script>\n</body>\n</html>\n'


# ── app + routes ───────────────────────────────────────────────────────────────
app = FastAPI(title="PitchCoach")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB (Groq free-tier audio file limit)


def _require_env(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        raise HTTPException(status_code=500, detail=f"Server missing env var: {name}")
    return val


async def _transcribe(audio_bytes: bytes, filename: str) -> str:
    # Groq hosts Whisper Large V3 behind an OpenAI-compatible API.
    api_key = _require_env("GROQ_API_KEY")
    client = openai.OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    buf = io.BytesIO(audio_bytes)
    buf.name = filename
    result = client.audio.transcriptions.create(model="whisper-large-v3", file=buf)
    return result.text.strip()


async def _embed(text: str) -> list:
    hf_token = _require_env("HF_API_KEY")
    model_id = _require_env("HF_MODEL_ID")
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 503:
            resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    return pool_hf_embedding(data)


def _classify(embedding: list, features: dict) -> dict:
    ling = features_to_vector(features)
    vector = list(embedding) + list(ling)
    if len(vector) != EMBEDDING_DIM + len(ling):
        raise HTTPException(status_code=500, detail="Feature vector size mismatch.")
    good_prob = positive_prob(score(vector))
    bad_prob = 1.0 - good_prob
    return {
        "model": "xgboost",
        "label": "good" if good_prob >= 0.5 else "bad",
        "good_prob": round(good_prob, 4),
        "bad_prob": round(bad_prob, 4),
    }


def _llm_feedback(transcript: str, classification: dict) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return ""
    label = classification["label"]
    good_pct = int(classification["good_prob"] * 100)
    bad_pct = int(classification["bad_prob"] * 100)
    feat = classification.get("text_features", {})
    feature_block = f"""
Model output (XGBoost on BERT embeddings + linguistic features)
  - Verdict          : {label.upper()} ({good_pct}% good / {bad_pct}% bad)

Linguistic analysis
  - Words            : {feat.get("word_count", "?")}
  - Sentences        : {feat.get("sentence_count", "?")}  (avg {feat.get("avg_sentence_length", "?")} words/sentence)
  - Filler words     : {feat.get("filler_word_count", 0)} found - {", ".join(feat.get("filler_words_found", [])) or "none"}
  - Hedging phrases  : {", ".join(feat.get("hedge_phrases_found", [])) or "none"}
  - Data / numbers   : {", ".join(feat.get("data_points", [])) or "none mentioned"}
  - Questions asked  : {feat.get("question_count", 0)}"""
    prompt = f"""You are an expert pitch coach who has studied the world's most influential speeches.

A user recorded a business pitch. An XGBoost classifier (trained on DistilBERT embeddings +
engineered linguistic features from 83 labelled entrepreneur pitches) scored it below, along
with a linguistic analysis of the transcript. Use this evidence to ground every piece of
feedback you give.

{feature_block}

Transcript:
\"\"\"{transcript}\"\"\"

Give concise, actionable coaching feedback in 4 sections:

1. **Model verdict explained** - in 1-2 sentences, explain what the model's score reveals
   about why it classified this pitch as {label}.

2. **Strengths** - 2-3 bullet points on what worked well, referencing specific words or
   moments from the transcript where relevant.

3. **Areas to improve** - 2-3 bullet points on specific weaknesses with concrete fixes,
   calling out any filler words, hedging, missing data points, or structural gaps flagged.

4. **One key takeaway** - the single most impactful change the speaker should make.

Keep the tone encouraging but honest. Be specific to the evidence above."""
    return generate_gemini_text(api_key, prompt, max_output_tokens=2048)


@app.get("/", response_class=HTMLResponse)
async def home():
    return _INDEX_HTML


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    import pathlib
    ext = pathlib.Path(audio.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use: {', '.join(ALLOWED_SUFFIXES)}")
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large ({len(audio_bytes)//1024//1024} MB). Max 25 MB.")
    try:
        transcript = await _transcribe(audio_bytes, audio.filename or f"audio{ext}")
    except openai.OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"Transcription (Groq) error: {e}")
    if not transcript:
        raise HTTPException(status_code=422, detail="Transcription returned empty text.")
    features = extract_text_features(transcript)
    embedding = await _embed(clean_text(transcript))
    classification = _classify(embedding, features)
    classification["text_features"] = features
    feedback = _llm_feedback(transcript, classification)
    return {
        "transcript": transcript,
        "word_count": len(transcript.split()),
        "classification": classification,
        "feedback": feedback,
    }
