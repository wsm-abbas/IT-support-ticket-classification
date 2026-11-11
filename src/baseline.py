"""TF-IDF + logistic regression baseline."""
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main():
    df = pd.read_csv("data/tickets.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2, random_state=42, stratify=df["category"]
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True),
        LogisticRegression(max_iter=1000),
    )

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)
    joblib.dump(model, "models/tfidf_logreg.joblib")
    print(classification_report(y_test, preds))

    results = {
        "model": "tfidf_logreg",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
        "train_seconds": round(train_time, 1),
    }
    print(results)
    with open("results/baseline.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
