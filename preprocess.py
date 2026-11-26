"""
preprocess.py
-------------
Text cleaning shared by training, the CLI and the Streamlit app.

Keeping this in ONE module guarantees the exact same cleaning is applied
at training time and at prediction time (a common source of silent bugs).
"""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MONEY_RE = re.compile(r"[$£€₹]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*\s?(?:usd|gbp|eur|inr|dollars|pounds)\b", re.I)
PHONE_RE = re.compile(r"\b\+?\d[\d\-\s]{8,}\d\b")
NUMBER_RE = re.compile(r"\d+")
HTML_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lower-case a message and replace noisy tokens with placeholders.

    URLs, e-mail addresses, money amounts, phone numbers and other numbers are
    strong spam signals, but their exact values are not. Replacing them with
    placeholder words (urltoken, moneytoken, ...) lets the model learn
    "this message contains a URL" instead of memorising specific links.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    text = text.lower()
    return SPACE_RE.sub(" ", text).strip()


def extra_signals(text: str) -> dict:
    """Simple human-readable red flags shown in the UI (not used by the model)."""
    t = text if isinstance(text, str) else ""
    letters = [c for c in t if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    low = t.lower()
    urgency = ["urgent", "immediately", "act now", "limited time", "expires", "final notice", "last chance", "within 24 hours"]
    return {
        "links": len(URL_RE.findall(t)),
        "money mentions": len(MONEY_RE.findall(t)),
        "exclamation marks": t.count("!"),
        "ALL-CAPS ratio": round(caps_ratio, 2),
        "urgency phrases": sum(p in low for p in urgency),
    }
