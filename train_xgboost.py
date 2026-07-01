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
    pitch_coach_xgb.json    — native XGBoost model (for the local Streamlit app)
    model_trees.json        — tree dump read by the pure-Python evaluator on Vercel
"""

import os

# macOS: torch and xgboost both ship libomp; loading both can crash the process.
# These must be set before importing torch / xgboost.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
    positive_prob,
)

# ── constants ────────────────────────────────────────────────────────────────
BERT_PATH = "pitch_coach_model"
ROOT_OUT = "pitch_coach_xgb.json"        # native model, for the local Streamlit app
TREES_OUT = "model_trees.json"           # tree dump, read by model_predict.py (root)
MAX_LEN = 512
SEED = 42
N_FOLDS = 5  # stratified k-fold cross-validation

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── pure-Python tree export ───────────────────────────────────────────────────
def export_trees_json(clf, out_path: str) -> None:
    """
    Dump the fitted XGBoost model to a compact JSON the pure-Python evaluator
    (model_predict.py) can walk without the xgboost library.

    Format: {"base_margin": float, "trees": [{nodeid: node, ...}, ...]}
    where each node is either {"leaf": v} or
    {"f": feature_index, "thr": split, "yes": id, "no": id, "missing": id}.
    """
    import json
    import math

    booster = clf.get_booster()
    booster.feature_names = None  # force split names to be "f<index>"
    dumps = booster.get_dump(dump_format="json")

    config = json.loads(booster.save_config())
    # xgboost 3.x reports base_score as a bracketed vector string, e.g. "[4.8E-1]".
    raw_bs = config["learner"]["learner_model_param"]["base_score"]
    base_score = float(str(raw_bs).strip().lstrip("[").rstrip("]"))
    # base_score is in probability space for binary:logistic → convert to margin.
    base_margin = math.log(base_score / (1.0 - base_score)) if 0 < base_score < 1 else 0.0

    def parse(node: dict, out: dict) -> None:
        nid = str(node["nodeid"])
        if "leaf" in node:
            out[nid] = {"leaf": node["leaf"]}
        else:
            split = node["split"]
            fidx = int(split[1:]) if isinstance(split, str) and split[0] == "f" else int(split)
            out[nid] = {
                "f": fidx,
                "thr": node["split_condition"],
                "yes": node["yes"],
                "no": node["no"],
                "missing": node["missing"],
            }
            for child in node["children"]:
                parse(child, out)

    trees = []
    for d in dumps:
        nodes: dict = {}
        parse(json.loads(d), nodes)
        trees.append(nodes)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"base_margin": base_margin, "trees": trees}))


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
    print(f"Feature matrix: {X.shape}  ({EMBEDDING_DIM} emb + {ling.shape[1]} linguistic)")

    # Regularized config to curb overfitting on the small (83-row) dataset.
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

    clf.save_model(ROOT_OUT)
    print(f"Saved native XGBoost model to '{ROOT_OUT}'.")

    export_trees_json(clf, TREES_OUT)
    print(f"Exported trees to '{TREES_OUT}'.")

    # Sanity check: pure-Python evaluator must match xgboost's own probability.
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_predict", "model_predict.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    x0 = X[0].tolist()
    xgb_p = float(clf.predict_proba(X[:1])[0][1])
    py_p = positive_prob(mp.score(x0))
    print(f"Evaluator check — xgboost: {xgb_p:.4f}  pure-python: {py_p:.4f}  "
          f"(Δ={abs(xgb_p - py_p):.2e})")

    print(
        "\nNext steps:\n"
        "  • Local:  streamlit run app.py\n"
        "  • Vercel: commit model_trees.json and redeploy."
    )


if __name__ == "__main__":
    main()
