# 📧 Email / SMS Spam Detector

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-pytest-green)

An end-to-end supervised machine learning project that classifies messages as **spam** or **ham** (not spam).
It cleans the text, extracts TF-IDF features, compares three classifiers with cross-validation, saves the best
one, and serves it through a **Streamlit web app** and a **command-line tool**.

## ✨ Features

- **Real + synthetic data:** the public SMS Spam Collection plus generated email-style messages (subject + body)
- **Smart text cleaning:** URLs, email addresses, money amounts and phone numbers become placeholder tokens, so the model learns "has a link" instead of memorising specific links
- **Model comparison:** Naive Bayes, Logistic Regression and a calibrated Linear SVM, selected by 5-fold cross-validation
- **Leak-free evaluation:** the train/test split happens before the vectorizer is fitted
- **Web app:** sample messages, adjustable spam threshold, red-flag counters and a performance panel
- **CLI:** classify one message, a text file, or a whole CSV
- **Explainability:** a plot of the most spam-like and ham-like words
- **Tests:** pytest checks for the cleaning step and model sanity

## 📊 Results

Held-out test set (1,212 messages), selected model: calibrated Linear SVM.

| Metric | Score |
|---|---|
| Accuracy | 99.1% |
| Spam precision | 98.1% |
| Spam recall | 96.8% |
| Spam F1 | 97.5% |

<p align="center">
  <img src="reports/confusion_matrix.png" width="320" alt="Confusion matrix">
  <img src="reports/top_features.png" width="520" alt="Top spam and ham terms">
</p>

> **Note:** on the real SMS data alone, spam F1 is about 0.96. The synthetic emails are template-based and
> score close to 100%, so they show the pipeline works but don't prove performance on real-world inboxes.

## 🗂️ Project structure

```
spam_detector/
├── app.py                     # Streamlit web app
├── predict.py                 # Command-line predictions
├── train_model.py             # Train, compare, evaluate, save
├── preprocess.py              # Shared text cleaning + red-flag heuristics
├── generate_dataset.py        # Builds the synthetic email dataset
├── requirements.txt
├── data/
│   ├── sms_spam.csv           # SMS Spam Collection (real data)
│   └── emails_synthetic.csv   # Generated email-style messages
├── models/
│   └── spam_model.pkl         # Saved scikit-learn Pipeline
├── reports/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   └── top_features.png
└── tests/
    └── test_pipeline.py
```

## 🚀 Getting started

**1. Clone and install**

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. (Optional) Train the model.** A trained model is already included.

```bash
python train_model.py
```

**3. Run the web app**

```bash
streamlit run app.py
```

Paste a message, click **Check message**, and see the verdict, spam probability and red flags.

## 💻 Command-line usage

```bash
# Single message
python predict.py "Congratulations! You won $1000. Claim now http://free-prize.win"
# -> SPAM  (spam probability: 99.x%)

# From a text file
python predict.py --file email.txt

# Score a whole CSV (needs a `message` column)
python predict.py --csv inbox.csv --out scored.csv
```

## 🧠 How it works

1. **Load** one or more CSVs with `message` and `label` (spam/ham) columns and drop duplicates
2. **Split** into stratified 80/20 train and test sets
3. **Clean** the text (lower-case, strip HTML, replace URLs, emails, money, phones and numbers with tokens)
4. **Vectorize** with TF-IDF (unigrams + bigrams, English stop-words, sublinear TF)
5. **Compare** the three classifiers with 5-fold cross-validation (macro F1)
6. **Evaluate** the winner on the untouched test set, overall and per data source
7. **Save** the full pipeline with `joblib`, so prediction applies exactly the same preprocessing as training

## 🔧 Train on your own data

Add a CSV with `message` and `label` columns (values `spam` / `ham`), then run:

```bash
python train_model.py --data data/sms_spam.csv data/my_emails.csv
```

Good real-world email corpora: Enron-Spam, the SpamAssassin Public Corpus, or the Kaggle "Spam Email" datasets.
To regenerate the synthetic emails: `python generate_dataset.py --n 3000 --seed 7`.

## ✅ Tests

```bash
python -m pytest -q tests
```

## ⚠️ Limitations

- Bag-of-words models can't understand context, images or attachments, and accuracy drops as spam tactics change, so retrain periodically
- The SMS data is short text from roughly 2005 to 2010; modern email spam looks different
- Treat the output as a filtering aid, not a security guarantee, and never click links in suspicious messages

## 🔮 Ideas for improvement

- Train on a larger real email corpus (Enron-Spam, SpamAssassin)
- Try transformer embeddings (e.g. DistilBERT) and compare against the TF-IDF baseline
- Parse raw `.eml` files (headers, sender domain, attachments) as extra features
- Add a feedback button in the app to collect corrections for retraining
