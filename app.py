"""
app.py - Streamlit web UI for the spam detector.

    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocess import extra_signals  # noqa: F401  (clean_text is needed to unpickle the model)
import preprocess  # noqa: F401

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "spam_model.pkl"
METRICS_PATH = ROOT / "reports" / "metrics.json"

SAMPLES = {
    "Prize scam": "Subject: You have WON $10,000!!!\nCongratulations! You were selected as the lucky winner. Claim your prize now: http://claim-now.win/x . Offer expires in 24 hours!",
    "Phishing": "URGENT: Your account will be suspended unless you verify your details immediately at http://paypa1-support.co/verify",
    "Work email": "Subject: Meeting moved to Thursday\nHi Priya, can we move the sprint retro sync to Thursday at 3pm? Let me know if that works for you. Thanks!",
    "Friendly note": "Hey, are we still meeting for lunch tomorrow at 1pm? There's a new place near the office I want to try.",
}

st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model = load_model()
st.title("📧 Spam Detector")
st.caption("TF-IDF text features + a classifier trained on real SMS spam and e-mail style messages")

if model is None:
    st.error("No trained model found. Run `python train_model.py` first.")
    st.stop()

st.session_state.setdefault("text", "")
cols = st.columns(len(SAMPLES))
for col, (label, txt) in zip(cols, SAMPLES.items()):
    if col.button(label):
        st.session_state["text"] = txt

text = st.text_area("Message / e-mail text", key="text", height=200,
                    placeholder="Paste an email or message here...")
threshold = st.slider("Spam threshold", 0.1, 0.9, 0.5, 0.05,
                      help="Flag as spam when the spam probability is at least this value. Lower = catches more spam but more false alarms.")

if st.button("Check message", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        spam_idx = list(model.classes_).index("spam")
        p_spam = float(model.predict_proba([text])[0][spam_idx])
        if p_spam >= threshold:
            st.error(f"🚫 **SPAM** - {p_spam:.1%} spam probability")
        else:
            st.success(f"✅ **NOT SPAM (ham)** - {p_spam:.1%} spam probability")
        st.progress(min(max(p_spam, 0.0), 1.0))

        st.subheader("Red flags in the text")
        sig = extra_signals(text)
        c = st.columns(len(sig))
        for col, (k, v) in zip(c, sig.items()):
            col.metric(k, v)
        st.caption("These are simple heuristics shown for context; the verdict comes from the trained model.")

with st.expander("Model performance"):
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        st.write(f"**Selected model:** {m['best_model']}  |  trained on {m['n_train']} / tested on {m['n_test']} messages")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.1%}"); b.metric("Spam precision", f"{m['spam_precision']:.1%}")
        c.metric("Spam recall", f"{m['spam_recall']:.1%}"); d.metric("Spam F1", f"{m['spam_f1']:.1%}")
        st.dataframe(pd.DataFrame(m["model_comparison"]).T)
        for img in ("confusion_matrix.png", "top_features.png"):
            if (ROOT / "reports" / img).exists():
                st.image(str(ROOT / "reports" / img))
    else:
        st.info("Run `python train_model.py` to generate metrics.")
