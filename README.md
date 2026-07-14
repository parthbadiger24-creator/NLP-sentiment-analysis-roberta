# IMDb Sentiment Analysis with RoBERTa

> Fine-tuned **RoBERTa** for binary sentiment classification on IMDb movie reviews, benchmarked against classical TF-IDF baselines (Logistic Regression + Multinomial Naive Bayes).
>
> **SSIM916 — Machine Learning for Social Data Science, University of Exeter**

---

## 🎯 Research Question

Can a transfer-learning model like RoBERTa accurately classify IMDb movie reviews as positive or negative — and does it beat classical TF-IDF baselines on a small labelled sample?

## 📦 Dataset

- **Source:** IMDb Movie Reviews (50,000 labelled reviews, balanced 25k / 25k)
- **Working sample:** **1,000 reviews** (stratified 800 train / 200 test) — chosen to make transformer fine-tuning tractable on the available compute
- **Labels:** `1 = positive`, `0 = negative`

> ⚠️ **Honest scope:** the fine-tuning + evaluation uses a 1,000-row subset, not the full 50k. Results below should be read as a proof-of-concept comparison, not a leaderboard score.

## 🧪 Methodology

| Model | Vectoriser | Training details |
|---|---|---|
| **RoBERTa (base)** | `roberta-base` tokenizer | 4 epochs · LR 1e-5 · batch 16 · AdamW · early stopping · `simpletransformers` |
| Logistic Regression | TF-IDF | scikit-learn baseline |
| Multinomial Naive Bayes | TF-IDF | scikit-learn baseline |

## 📈 Results

| Model | Test Accuracy |
|---|---|
| **RoBERTa (fine-tuned)** | **0.9050** |
| Logistic Regression + TF-IDF | 0.8150 |
| Multinomial Naive Bayes + TF-IDF | 0.7700 |

Precision / recall / F1 also reported per class in the notebook. Transfer learning delivers a **~9pp accuracy gain over the strongest classical baseline** on this sample.

## 🧰 Tech Stack

`Python` · `PyTorch` · `HuggingFace transformers` · `simpletransformers` · `scikit-learn` · `pandas` · `wordcloud` · `matplotlib`

## 📁 Repo Structure

```text
.
├── notebooks/
│   └── roberta_imdb_sentiment.ipynb
├── docs/
│   └── IMDB_RoBERTa_Report.pdf
├── requirements.txt
└── README.md
```

## ▶️ Reproduce

```bash
pip install -r requirements.txt
jupyter lab notebooks/roberta_imdb_sentiment.ipynb
```

## 📄 Report

Full write-up → [`docs/IMDB_RoBERTa_Report.pdf`](docs/IMDB_RoBERTa_Report.pdf)

---

<sub>MIT-licensed · Author: [Parth Badiger](https://github.com/parthbadiger24-creator)</sub>
