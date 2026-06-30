"""
predictor.py — Whisper transcription + DistilBERT inference helpers.

Following the pattern from the article:
  https://wildangunawan.medium.com/bert-serverless-deployment-with-streamlit-and-its-free-5d9f20154f24

All heavy objects are cached with @st.cache_resource so they load once per session.
"""

import os

# macOS: torch and xgboost both ship libomp; loading both can crash the process.
# These must be set before importing torch / xgboost.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import shutil
import tempfile
from pathlib import Path

import streamlit as st
import torch

from feature_utils import (
    EMBEDDING_DIM,
    clean_text,
    extract_text_features,
    features_to_vector,
)

def _ensure_ffmpeg_on_path() -> None:
    """Whisper shells out to `ffmpeg`; ensure Homebrew's bin dir is on PATH."""
    if shutil.which("ffmpeg"):
        return
    for brew_bin in ("/opt/homebrew/bin", "/usr/local/bin"):
        ffmpeg = Path(brew_bin) / "ffmpeg"
        if ffmpeg.is_file():
            os.environ["PATH"] = f"{brew_bin}{os.pathsep}{os.environ.get('PATH', '')}"
            return
    raise RuntimeError(
        "ffmpeg is required for audio transcription but was not found.\n\n"
        "Install on macOS:\n"
        "  brew install ffmpeg\n\n"
        "Then restart Streamlit:\n"
        "  streamlit run app.py"
    )


# ── model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Whisper …")
def load_whisper(model_size: str = "base"):
    import whisper

    # Common pitfall: a different pip package named "whisper" gets installed,
    # which does not provide `load_model`. We need OpenAI's "openai-whisper".
    if not hasattr(whisper, "load_model"):
        mod_path = getattr(whisper, "__file__", "<unknown>")
        raise RuntimeError(
            "Whisper backend is not OpenAI Whisper (missing whisper.load_model).\n"
            f"Imported module path: {mod_path}\n\n"
            "Fix (run inside your project env):\n"
            "  pip uninstall -y whisper\n"
            "  pip install -U openai-whisper\n"
        )

    _ensure_ffmpeg_on_path()
    return whisper.load_model(model_size)


@st.cache_resource(show_spinner="Loading BERT classifier …")
def load_bert(model_path: str = "pitch_coach_model"):
    from transformers import AutoTokenizer, DistilBertForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # eager attention is required for output_attentions=True (used for the
    # "top attended words" explainability); SDPA silently returns no weights.
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path, attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner="Loading XGBoost model …")
def load_xgb(model_path: str = "pitch_coach_xgb.json"):
    import xgboost as xgb

    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    return clf


# ── transcription ─────────────────────────────────────────────────────────────

def transcribe(audio_bytes: bytes, whisper_model, suffix: str = ".wav") -> str:
    """Write audio bytes to a temp file, transcribe, return transcript text."""
    _ensure_ffmpeg_on_path()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        result = whisper_model.transcribe(tmp_path, fp16=False)
    except FileNotFoundError as exc:
        if "ffmpeg" in str(exc).lower():
            raise RuntimeError(
                "ffmpeg is required for audio transcription but was not found.\n\n"
                "Install on macOS:\n"
                "  brew install ffmpeg\n\n"
                "Then restart Streamlit:\n"
                "  streamlit run app.py"
            ) from exc
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result["text"].strip()


# ── BERT embedding + attention helpers ────────────────────────────────────────

def _top_attended_words(outputs, inputs, tokenizer) -> list:
    """Words BERT weighted most heavily (CLS attention), for explainability."""
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        return []

    stacked = torch.stack(list(attentions))   # (layers, batch, heads, L, L)
    avg_attn = stacked.mean(dim=(0, 1, 2))     # (L, L)
    cls_attn = avg_attn[0]                      # CLS row → (L,)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_scores: dict[str, float] = {}
    current_word, current_score = "", 0.0
    skip = {"[CLS]", "[SEP]", "[PAD]"}

    for token, score in zip(tokens, cls_attn.tolist()):
        if token in skip:
            continue
        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
        else:
            if current_word:
                word_scores[current_word] = word_scores.get(current_word, 0) + current_score
            current_word, current_score = token, score
    if current_word:
        word_scores[current_word] = word_scores.get(current_word, 0) + current_score

    stop = {"a", "an", "the", "is", "it", "in", "of", "to", "and", "or", "i"}
    ranked = sorted(
        [(w, s) for w, s in word_scores.items() if len(w) > 1 and w not in stop],
        key=lambda x: x[1],
        reverse=True,
    )
    return [w for w, _ in ranked[:12]]


def _embed(outputs, inputs) -> "torch.Tensor":
    """Mean-pool DistilBERT's last hidden state into one 768-d embedding."""
    hidden = outputs.hidden_states[-1][0]             # (tokens, 768)
    mask = inputs["attention_mask"][0].unsqueeze(-1)  # (tokens, 1)
    return (hidden * mask).sum(0) / mask.sum().clamp(min=1)


# ── classification (XGBoost on BERT embedding + linguistic features) ───────────

def classify_xgb(text, bert_model, tokenizer, xgb_model, max_len: int = 512) -> dict:
    """
    Runs the full XGBoost pipeline:
        BERT embedding (768) + linguistic features (8) → XGBoost → good/bad

    Returns:
        {
            "model": "xgboost",
            "label": "good" | "bad",
            "good_prob": float,
            "bad_prob": float,
            "top_attended_words": list[str],   # BERT explainability signal
            "text_features": dict,             # linguistic signals
        }
    """
    import numpy as np

    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True, max_length=max_len
    )

    with torch.no_grad():
        outputs = bert_model(
            **inputs, output_attentions=True, output_hidden_states=True
        )

    embedding = _embed(outputs, inputs).cpu().numpy()          # (768,)
    assert embedding.shape[0] == EMBEDDING_DIM

    feats = extract_text_features(text)
    ling = np.array(features_to_vector(feats), dtype="float32")  # (8,)

    X = np.hstack([embedding, ling]).astype("float32").reshape(1, -1)
    good_prob = float(xgb_model.predict_proba(X)[0][1])
    bad_prob = 1.0 - good_prob
    label = "good" if good_prob >= 0.5 else "bad"

    return {
        "model": "xgboost",
        "label": label,
        "good_prob": round(good_prob, 4),
        "bad_prob": round(bad_prob, 4),
        "top_attended_words": _top_attended_words(outputs, inputs, tokenizer),
        "text_features": feats,
    }


# ── LLM feedback (optional, requires GOOGLE_API_KEY) ─────────────────────────

from gemini_utils import format_gemini_error, generate_gemini_text


def get_llm_feedback(transcript: str, classification: dict) -> str:
    """
    Calls Google Gemini to give actionable pitch coaching feedback.
    Uses all BERT outputs — probabilities, attention-based features, and
    linguistic signals — to ground the feedback in model evidence.
    Returns an empty string if the API key is not set.
    """
    import os
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return ""

    label       = classification["label"]
    good_pct    = int(classification["good_prob"] * 100)
    bad_pct     = int(classification["bad_prob"]  * 100)
    top_words   = classification.get("top_attended_words", [])
    feat        = classification.get("text_features", {})

    # Format feature block so Gemini gets structured evidence
    feature_block = f"""
Model outputs (XGBoost trained on BERT embeddings + linguistic features)
  • Verdict          : {label.upper()} ({good_pct}% good / {bad_pct}% bad)
  • Top attended words (words BERT's embedding weighted most):
      {", ".join(top_words) if top_words else "n/a"}

Linguistic analysis
  • Words            : {feat.get("word_count", "?")}
  • Sentences        : {feat.get("sentence_count", "?")}  (avg {feat.get("avg_sentence_length", "?")} words/sentence)
  • Filler words     : {feat.get("filler_word_count", 0)} found — {", ".join(feat.get("filler_words_found", [])) or "none"}
  • Hedging phrases  : {", ".join(feat.get("hedge_phrases_found", [])) or "none"}
  • Data / numbers   : {", ".join(feat.get("data_points", [])) or "none mentioned"}
  • Questions asked  : {feat.get("question_count", 0)}"""

    prompt = f"""You are an expert pitch coach who has studied the world's most influential speeches.

A user recorded a business pitch. Below are the outputs from an XGBoost classifier trained on
DistilBERT embeddings combined with engineered linguistic features (built from 83 labelled
entrepreneur pitches), along with a linguistic analysis of the transcript.
Use this evidence to ground every piece of feedback you give.

{feature_block}

Transcript:
\"\"\"{transcript}\"\"\"

Give concise, actionable coaching feedback in 4 sections:

1. **Model verdict explained** — in 1-2 sentences, explain what the model's score and the
   top attended words reveal about why it classified this pitch as {label}.

2. **Strengths** — 2-3 bullet points on what worked well, referencing specific words or
   moments from the transcript where relevant.

3. **Areas to improve** — 2-3 bullet points on specific weaknesses with concrete fixes,
   calling out any filler words, hedging, missing data points, or structural gaps
   the analysis flagged.

4. **One key takeaway** — the single most impactful change the speaker should make.

Keep the tone encouraging but honest. Be specific to the evidence above."""

    return generate_gemini_text(api_key, prompt, max_output_tokens=2048)
