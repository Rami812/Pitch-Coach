"""
feature_utils.py — shared feature engineering for the XGBoost pitch classifier.

This module is the SINGLE SOURCE OF TRUTH for:
  • linguistic feature extraction (used in training + both inference paths)
  • the exact numeric feature vector ordering fed to XGBoost
  • pooling raw BERT hidden states into one 768-d embedding

Keeping all of this in one dependency-light module (no torch / no streamlit)
means the local app, the training script, and the Vercel function all produce
identical feature vectors — otherwise XGBoost would see mismatched inputs.
"""

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


def pool_hf_embedding(hf_response) -> list:
    """
    Mean-pool a HuggingFace feature-extraction response into one 768-d vector.

    The feature-extraction pipeline returns token-level embeddings whose shape
    can be (tokens, 768), (1, tokens, 768), or already-pooled (768,). We reduce
    whatever we get down to a single 768-length list.
    """
    import numpy as np

    arr = np.array(hf_response, dtype="float32")
    while arr.ndim > 2:          # e.g. (1, tokens, 768) -> (tokens, 768)
        arr = arr[0]
    if arr.ndim == 2:            # (tokens, 768) -> (768,)
        arr = arr.mean(axis=0)
    # arr is now 1-D
    if arr.shape[0] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding size {arr.shape[0]}, expected {EMBEDDING_DIM}."
        )
    return arr.tolist()
