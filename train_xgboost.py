"""
Train an XGBoost pitch classifier on BERT embeddings + linguistic features.

Pipeline:
    transcript
      → DistilBERT (mean-pooled last hidden state)  → 768-d embedding
      → linguistic features                          → 8-d feature vector
      → concatenate                                  → 776-d input
      → XGBoost binary classifier                    → good / bad

Run AFTER train_bert.py (which produces pitch_coach_model/):
    python train_xgboost.py

Outputs:
    pitch_coach_xgb.json       — XGBoost model (for the local Streamlit app)
    api/pitch_coach_xgb.json   — copy bundled into the Vercel function
"""

import os

# macOS: torch and xgboost both ship libomp; loading both can crash the process.
# These must be set before importing torch / xgboost.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import AutoTokenizer, DistilBertModel

from feature_utils import (
    EMBEDDING_DIM,
    LINGUISTIC_FEATURE_NAMES,
    clean_text,
    extract_text_features,
    features_to_vector,
)

# ── constants ────────────────────────────────────────────────────────────────
BERT_PATH = "pitch_coach_model"
ROOT_OUT = "pitch_coach_xgb.json"
API_OUT = "api/pitch_coach_xgb.json"
MAX_LEN = 512
SEED = 42
N_FOLDS = 5  # stratified k-fold cross-validation

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── BERT embedding ────────────────────────────────────────────────────────────
def embed_corpus(texts, tokenizer, model) -> np.ndarray:
    """Mean-pool DistilBERT's last hidden state into one 768-d vector per text."""
    vectors = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                clean_text(text),
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
            ).to(device)
            hidden = model(**inputs).last_hidden_state[0]  # (tokens, 768)
            mask = inputs["attention_mask"][0].unsqueeze(-1)  # (tokens, 1)
            pooled = (hidden * mask).sum(0) / mask.sum().clamp(min=1)
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def main() -> None:
    if not Path(BERT_PATH).exists():
        raise SystemExit(
            f"BERT model not found at '{BERT_PATH}/'. Run `python train_bert.py` first."
        )

    print("Loading dataset …")
    df = pd.read_excel("Final_Dataset.xlsx")
    df["text"] = df["Speech"].astype(str)
    y = (df["label"] == "good").astype(int).values  # 1=good, 0=bad

    # Load DistilBERT as a *base* model (no classification head) for embeddings.
    print(f"Loading BERT from '{BERT_PATH}' …")
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
    bert = DistilBertModel.from_pretrained(BERT_PATH).to(device)

    print("Computing BERT embeddings …")
    embeddings = embed_corpus(df["text"].tolist(), tokenizer, bert)
    assert embeddings.shape[1] == EMBEDDING_DIM

    print("Extracting linguistic features …")
    ling = np.array(
        [features_to_vector(extract_text_features(t)) for t in df["text"]],
        dtype="float32",
    )

    X = np.hstack([embeddings, ling]).astype("float32")
    feature_names = [f"emb_{i}" for i in range(EMBEDDING_DIM)] + LINGUISTIC_FEATURE_NAMES
    print(f"Feature matrix: {X.shape}  ({EMBEDDING_DIM} emb + {ling.shape[1]} linguistic)")

    # Regularized config — shallow trees + heavy subsampling + L1/L2 to curb the
    # overfitting that a single 80/20 split on 83 rows masks. The factory returns
    # a fresh estimator for each CV fold and for the final fit.
    def make_clf():
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            min_child_weight=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.5,
            eval_metric="logloss",
            random_state=SEED,
        )

    # ── stratified k-fold cross-validation (honest generalization estimate) ────
    # cross_val_predict trains on N-1 folds and predicts the held-out fold, so
    # every row gets an out-of-fold prediction it never trained on.
    print(f"\nRunning {N_FOLDS}-fold stratified cross-validation …")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_pred = cross_val_predict(make_clf(), X, y, cv=skf)

    fold_accs = []
    for k, (_, test_idx) in enumerate(skf.split(X, y), 1):
        a = accuracy_score(y[test_idx], oof_pred[test_idx])
        fold_accs.append(a)
        print(f"  fold {k}: acc={a:.3f}  (n={len(test_idx)})")

    cv_acc = accuracy_score(y, oof_pred)
    print(f"\nCross-validated accuracy: {cv_acc:.4f}  "
          f"(mean fold {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f})")
    print("Out-of-fold classification report:")
    print(classification_report(y, oof_pred, target_names=["bad", "good"]))

    # ── final model: fit on ALL data and save ──────────────────────────────────
    print("Fitting final model on all data …")
    clf = make_clf()
    clf.fit(X, y)
    clf.get_booster().feature_names = feature_names  # readable names for SHAP

    clf.save_model(ROOT_OUT)
    Path(API_OUT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT_OUT, API_OUT)
    print(f"Saved XGBoost model to '{ROOT_OUT}' and '{API_OUT}'.")
    print(
        "\nNext steps:\n"
        "  • Local:  streamlit run app.py\n"
        "  • Vercel: redeploy — api/pitch_coach_xgb.json is bundled automatically."
    )


if __name__ == "__main__":
    main()
