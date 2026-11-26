"""
train_model.py
--------------
Trains and compares several classifiers for spam detection, keeps the best
one, evaluates it on a held-out test set and saves everything needed by the
app.

    python train_model.py
    python train_model.py --data data/sms_spam.csv            # a single dataset
    python train_model.py --data a.csv b.csv --test-size 0.25 # several datasets

Every CSV needs the columns `message` and `label` (spam / ham).
A `source` column is optional and is only used for per-source reporting.

Outputs
    models/spam_model.pkl          sklearn Pipeline (cleaning + TF-IDF + classifier)
    reports/metrics.json           all scores + model comparison
    reports/confusion_matrix.png   test-set confusion matrix
    reports/top_features.png       words that push a message toward spam / ham
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

DEFAULT_DATA = ["data/sms_spam.csv", "data/emails_synthetic.csv"]
MODEL_PATH = Path("models/spam_model.pkl")
REPORT_DIR = Path("reports")
SEED = 42


def load_data(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "source" not in df.columns:
            df["source"] = Path(p).stem
        frames.append(df[["message", "label", "source"]])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["message", "label"])
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])].drop_duplicates(subset="message")
    return df.reset_index(drop=True)


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def candidates():
    return {
        "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.1)),
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=2000, C=10, class_weight="balanced")),
        "Linear SVM (calibrated)": make_pipeline(CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
    }


def plot_top_features(pipe, path, k=15):
    """Show most spam-like / ham-like words (works for NB and linear models)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    if hasattr(clf, "feature_log_prob_"):
        classes = list(clf.classes_)
        score = clf.feature_log_prob_[classes.index("spam")] - clf.feature_log_prob_[classes.index("ham")]
    elif hasattr(clf, "coef_"):
        score = clf.coef_[0] if list(clf.classes_)[1] == "spam" else -clf.coef_[0]
    else:  # calibrated model: average the inner linear models
        coefs = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
        score = coefs if list(clf.classes_)[1] == "spam" else -coefs
    order = np.argsort(score)
    ham_idx, spam_idx = order[:k], order[-k:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh(names[spam_idx], score[spam_idx], color="#d9534f"); axes[0].set_title("Most SPAM-like terms")
    axes[1].barh(names[ham_idx], -score[ham_idx], color="#5cb85c"); axes[1].set_title("Most HAM-like terms")
    plt.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=DEFAULT_DATA)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    REPORT_DIR.mkdir(exist_ok=True); MODEL_PATH.parent.mkdir(exist_ok=True)

    print("=" * 64); print("EMAIL / SMS SPAM DETECTION - TRAINING"); print("=" * 64)
    df = load_data(args.data)
    print(f"\n[1/5] Loaded {len(df)} unique labelled messages")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())

    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        df["message"], df["label"], df["source"],
        test_size=args.test_size, random_state=SEED, stratify=df["label"] + df["source"])
    print(f"\n[2/5] Split -> train {len(X_train)} | test {len(X_test)}")

    print("\n[3/5] Comparing models (5-fold CV on the training set, metric = macro F1)")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    comparison = {}
    for name, pipe in candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                                 scoring="f1_macro")
        comparison[name] = {"cv_f1_macro_mean": round(float(scores.mean()), 4),
                            "cv_f1_macro_std": round(float(scores.std()), 4)}
        print(f"      {name:<26} macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(comparison, key=lambda n: comparison[n]["cv_f1_macro_mean"])
    print(f"\n[4/5] Best model: {best_name} -> refitting on the full training set")
    best = candidates()[best_name]
    best.fit(X_train, y_train)

    y_pred = best.predict(X_test)
    metrics = {
        "best_model": best_name,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "spam_precision": round(precision_score(y_test, y_pred, pos_label="spam"), 4),
        "spam_recall": round(recall_score(y_test, y_pred, pos_label="spam"), 4),
        "spam_f1": round(f1_score(y_test, y_pred, pos_label="spam"), 4),
        "model_comparison": comparison,
        "per_source": {},
    }
    print("\n      Held-out TEST results")
    print(f"      accuracy={metrics['accuracy']}  spam precision={metrics['spam_precision']}  "
          f"recall={metrics['spam_recall']}  F1={metrics['spam_f1']}")
    print("\n" + classification_report(y_test, y_pred, digits=3))
    for s in sorted(src_test.unique()):
        m = (src_test == s).values
        metrics["per_source"][s] = {
            "n_test": int(m.sum()),
            "accuracy": round(accuracy_score(y_test[m], y_pred[m]), 4),
            "spam_f1": round(f1_score(y_test[m], y_pred[m], pos_label="spam"), 4)}
        print(f"      {s:<22} n={m.sum():<5} acc={metrics['per_source'][s]['accuracy']}  "
              f"spam F1={metrics['per_source'][s]['spam_f1']}")

    print("\n[5/5] Saving model + reports")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=["ham", "spam"], ax=ax, cmap="Blues")
    ax.set_title(f"Confusion matrix - {best_name}"); plt.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=130); plt.close(fig)
    plot_top_features(best, REPORT_DIR / "top_features.png")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(best, MODEL_PATH)
    print(f"      saved {MODEL_PATH}, {REPORT_DIR}/metrics.json, confusion_matrix.png, top_features.png")
    print("\nDone. Try:  python predict.py \"You won $1000! click http://x.win\"")


if __name__ == "__main__":
    main()
