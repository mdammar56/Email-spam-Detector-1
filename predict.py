"""
predict.py - classify messages from the command line.

    python predict.py "Congratulations, you won $1000! http://claim.win"
    python predict.py --file email.txt
    python predict.py --csv inbox.csv --out scored.csv      # needs a `message` column
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)


def score(model, texts):
    """Return (labels, spam_probabilities) for a list of texts."""
    spam_idx = list(model.classes_).index("spam")
    proba = model.predict_proba(texts)[:, spam_idx]
    labels = ["spam" if p >= 0.5 else "ham" for p in proba]
    return labels, proba


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="message text to classify")
    ap.add_argument("--file", help="read the message from a text file")
    ap.add_argument("--csv", help="classify every row of a CSV (column: message)")
    ap.add_argument("--out", help="where to write scored CSV (default: <csv>_scored.csv)")
    a = ap.parse_args()
    model = load_model()

    if a.csv:
        df = pd.read_csv(a.csv)
        df["prediction"], df["spam_probability"] = score(model, df["message"].fillna("").tolist())
        out = a.out or str(Path(a.csv).with_suffix("")) + "_scored.csv"
        df.to_csv(out, index=False)
        print(f"Scored {len(df)} messages -> {out}  ({(df.prediction == 'spam').sum()} flagged as spam)")
        return

    text = Path(a.file).read_text(errors="ignore") if a.file else a.text
    if not text:
        ap.error("give a message, --file or --csv")
    (label,), (p,) = score(model, [text])
    print(f"{label.upper()}  (spam probability: {p:.1%})")


if __name__ == "__main__":
    main()
