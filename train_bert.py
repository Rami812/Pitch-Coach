"""
Fine-tune DistilBERT on the PitchCoach speech quality dataset.

Usage:
    python train_bert.py

Outputs:
    pitch_coach_model/  -- saved HuggingFace model + tokenizer
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)


# ── constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "pitch_coach_model"
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 1
LR = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("", "")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ── data ──────────────────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_excel("Final_Dataset.xlsx")
df["text"] = df["Speech"].apply(clean_text)
df["label_id"] = (df["label"] == "good").astype(int)  # 1=good, 0=bad

texts = df["text"].tolist()
labels = df["label_id"].tolist()

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
train_dataset = SpeechDataset(X_train, y_train, tokenizer)
val_dataset = SpeechDataset(X_val, y_val, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# ── model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.config.id2label = {0: "bad", 1: "good"}
model.config.label2id = {"bad": 0, "good": 1}
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)


# ── training loop ─────────────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- train ---
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels
        )
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)

    # --- validate ---
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_labels.cpu().numpy())

    val_acc = np.mean(np.array(all_preds) == np.array(all_true))
    print(f"Epoch {epoch + 1}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        # save_pretrained only writes tokenizer.json; copy vocab.txt for older loaders
        vocab_src = getattr(tokenizer, "vocab_file", None)
        vocab_dest = Path(SAVE_PATH) / "vocab.txt"
        if vocab_src and not vocab_dest.exists():
            shutil.copy(vocab_src, vocab_dest)
        print(f"  ✓ saved best model (val_acc={val_acc:.4f})")

print("\nFinal validation report:")
print(classification_report(all_true, all_preds, target_names=["bad", "good"]))
print(f"\nBest model saved to '{SAVE_PATH}/' with val_acc={best_val_acc:.4f}")
