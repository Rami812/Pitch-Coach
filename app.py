"""
PitchCoach — Streamlit app
Upload or record a pitch → Whisper transcribes → DistilBERT classifies → Gemini coaches.

Run:
    streamlit run app.py

Before first run:
    python train_bert.py
    brew install ffmpeg   # required for Whisper audio decoding (macOS)
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load secrets from .env (gitignored) so GOOGLE_API_KEY is available locally.
load_dotenv()

from gemini_utils import format_gemini_error
from predictor import (
    classify_xgb,
    get_llm_feedback,
    load_bert,
    load_whisper,
    load_xgb,
    transcribe,
)

# This file must be run via `streamlit run app.py`. Running with `python app.py`
# results in "NoSessionContext" errors and broken caching/session state.
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        raise SystemExit(
            "This is a Streamlit app.\n\nRun:\n  streamlit run app.py\n"
        )
except Exception:
    # If Streamlit internals change, don't block startup; Streamlit will emit
    # its own warning when not run via `streamlit run`.
    pass

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchCoach",
    page_icon="🎙️",
    layout="centered",
)

# ── header ────────────────────────────────────────────────────────────────────
st.title("🎙️ PitchCoach")
st.markdown(
    """
**Analyze your business pitch against the world's most influential speeches.**

1. Upload an audio recording of your pitch (WAV, MP3, M4A, OGG, FLAC)
2. Whisper transcribes it automatically
3. An XGBoost model (BERT embeddings + linguistic features) scores it *good* or *bad*
4. (Optional) Gemini gives you actionable coaching feedback
"""
)
st.divider()

# ── sidebar — settings ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    whisper_size = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger = more accurate but slower. 'base' is a good default.",
    )
    bert_path = st.text_input(
        "BERT model path",
        value="pitch_coach_model",
        help="Directory created by train_bert.py",
    )
    xgb_path = st.text_input(
        "XGBoost model path",
        value="pitch_coach_xgb.json",
        help="File created by train_xgboost.py",
    )
    use_llm = st.toggle(
        "Gemini coaching feedback",
        value=bool(os.environ.get("GOOGLE_API_KEY")),
        help="Requires GOOGLE_API_KEY in your environment.",
    )
    st.divider()
    st.caption("Dataset: 83 labelled entrepreneur pitches (good / bad)")
    st.caption("Model: XGBoost on DistilBERT embeddings + linguistic features")

# ── check models exist ────────────────────────────────────────────────────────
if not Path(bert_path).exists():
    st.warning(
        f"No BERT model found at **{bert_path}/**. "
        "Run `python train_bert.py` first, then refresh this page.",
        icon="⚠️",
    )
    st.stop()

if not Path(xgb_path).exists():
    st.warning(
        f"No XGBoost model found at **{xgb_path}**. "
        "Run `python train_xgboost.py` first, then refresh this page.",
        icon="⚠️",
    )
    st.stop()

# ── load models (cached) ──────────────────────────────────────────────────────
whisper_model = load_whisper(whisper_size)
bert_model, tokenizer = load_bert(bert_path)
xgb_model = load_xgb(xgb_path)

# ── audio upload ──────────────────────────────────────────────────────────────
st.subheader("Upload your pitch")
uploaded = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload an audio file above to get started.", icon="⬆️")
    st.stop()

# ── display audio player ──────────────────────────────────────────────────────
st.audio(uploaded)

# Determine file extension for tempfile
ext_map = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "video/mp4": ".mp4",
}
suffix = ext_map.get(uploaded.type, Path(uploaded.name).suffix or ".wav")

# ── transcription ─────────────────────────────────────────────────────────────
st.subheader("Transcription")

with st.spinner("Transcribing audio with Whisper …"):
    transcript = transcribe(uploaded.getvalue(), whisper_model, suffix=suffix)

if not transcript:
    st.error("Whisper returned an empty transcript. Try a different audio file.")
    st.stop()

with st.expander("View full transcript", expanded=True):
    st.write(transcript)

word_count = len(transcript.split())
st.caption(f"{word_count} words transcribed")

st.divider()

# ── classification ────────────────────────────────────────────────────────────
st.subheader("Pitch quality assessment")

with st.spinner("Classifying with XGBoost (BERT embeddings + features) …"):
    result = classify_xgb(transcript, bert_model, tokenizer, xgb_model)

label = result["label"]
good_pct = int(result["good_prob"] * 100)
bad_pct = int(result["bad_prob"] * 100)

col1, col2 = st.columns([1, 2])

with col1:
    if label == "good":
        st.success("✅ Good pitch", icon="🏆")
    else:
        st.error("❌ Needs work", icon="📉")

with col2:
    st.metric("Good pitch confidence", f"{good_pct}%")
    st.metric("Bad pitch confidence", f"{bad_pct}%")

# confidence bar (green = good probability)
st.progress(result["good_prob"], text=f"Good pitch score: {good_pct}%")

# ── interpretation ────────────────────────────────────────────────────────────
st.divider()
st.subheader("What this means")

if label == "good":
    st.markdown(
        f"""
Your pitch scored **{good_pct}% good** — it shares characteristics with
influential speeches studied in our training set.

**Common traits of good pitches:**
- Clear problem + solution framing
- Confident, direct language with no filler hedging
- Concrete specifics (numbers, names, timelines)
- A compelling call to action
"""
    )
else:
    st.markdown(
        f"""
Your pitch scored **{bad_pct}% bad** — it shares traits with weaker pitches
in our training set.

**Common weaknesses to address:**
- Vague language or excessive jargon
- Lack of structure (problem → solution → traction → ask)
- Too many filler words or excessive hedging
- Missing a clear call to action
"""
    )

# ── LLM feedback ──────────────────────────────────────────────────────────────
if use_llm:
    st.divider()
    st.subheader("Gemini's coaching feedback")

    if not os.environ.get("GOOGLE_API_KEY"):
        st.warning(
            "Set the `GOOGLE_API_KEY` environment variable to enable this feature.",
            icon="🔑",
        )
    else:
        with st.spinner("Getting personalised coaching from Gemini …"):
            try:
                feedback = get_llm_feedback(transcript, result)
            except Exception as exc:
                st.error(format_gemini_error(exc))
                feedback = ""

        if feedback:
            st.markdown(feedback)
        else:
            st.warning("Could not retrieve feedback. Check your API key.")

# ── footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "PitchCoach · XGBoost on DistilBERT embeddings + linguistic features · "
    "Transcription by OpenAI Whisper · Coaching by Gemini"
)
