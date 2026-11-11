"""Fine-tune a small transformer (DistilBERT) on the same tickets.

Uses the same train/test split as baseline.py so the numbers can be
compared. It runs on CPU but is much happier on a GPU.
"""
import json
import time

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL_NAME = "distilbert-base-uncased"


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=64)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main():
    df = pd.read_csv("data/tickets.csv")
    classes = sorted(df["category"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for c, i in label2id.items()}
    y = df["category"].map(label2id)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=0.2, random_state=42, stratify=df["category"]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes), id2label=id2label, label2id=label2id
    )

    train_ds = TicketDataset(X_train, y_train, tokenizer)
    test_ds = TicketDataset(X_test, y_test, tokenizer)

    args = TrainingArguments(
        output_dir="results/transformer_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds)

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    output = trainer.predict(test_ds)
    preds = output.predictions.argmax(axis=1)
    print(classification_report(y_test, preds, target_names=classes))

    results = {
        "model": MODEL_NAME,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
        "train_seconds": round(train_time, 1),
    }
    print(results)
    with open("results/transformer.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
