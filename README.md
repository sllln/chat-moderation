# Real-Time Chat Moderation (Turkish)

SWE015 – Introduction to Large Language Models · Homework · **Use Case 4: Real-Time Chat Moderation**

A chat-like web app that labels every message **clean / offensive** in real time, masks offensive
messages and logs the response time of every message against a **50 ms budget on a CPU-only machine**.
The task is solved with a ladder of models, from a rule-based baseline to a quantized Transformer,
all evaluated on the same held-out test set.

- Demo video: _link will be added_
- Models on the Hugging Face Hub: https://huggingface.co/selliinn/chat-moderation-tr-models

---

## Quick start

Requirements: **Python 3.12** and Git. Works on Windows, macOS and Linux; no GPU needed.

**1. Get the code and create a virtual environment**

Windows (PowerShell):
```powershell
git clone https://github.com/sllln/chat-moderation.git
cd chat-moderation
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:
```bash
git clone https://github.com/sllln/chat-moderation.git
cd chat-moderation
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Run the app (single command, from the project folder)**
```bash
python src/app.py
```
On the first start the trained models (about 630 MB) are downloaded automatically from the
Hugging Face Hub. The app then opens in the browser at http://127.0.0.1:7860.

> If PowerShell says "running scripts is disabled", run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate the environment again.

---

## What the app does

- **Live chat:** every message is classified by the selected model (default: **BERTurk, INT8 quantized**),
  shown with a confidence score, and masked (`*****`) if it is offensive.
- **Model panel:** switch between the five models; a gauge shows the last response time against the
  50 ms budget, plus the average and the number of messages over budget.
- **Tabs:** *Log* (response time of every message), *Compare* (run all five models on the last message),
  *Reading* (the cleaned text each model actually sees and the lexicon words that matched).

---

## Results

Official OffensEval-TR 2020 **test set** (3,528 tweets, 20.3 % offensive). Latency = time to classify
**one** message, mean over 200 different test messages after a warm-up.
Hardware: 11th Gen Intel Core i7-11370H @ 3.30 GHz, **CPU only**, 4 PyTorch threads.

| Model | Numeric precision | Macro-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | ≤ 50 ms |
|---|---|---|---|---|---|---|
| Lexicon + edit distance | rules | 0.713 | 0.1 | 10.11 | 26.78 | yes |
| tf-idf + logistic regression | FP64 | 0.769 | 4.4 | 1.37 | 1.73 | yes |
| word2vec + feedforward NN | FP32 | 0.663 | 23.3 | 0.23 | 0.36 | yes |
| BERTurk | FP32 | 0.813 | 422.0 | 31.58 | 44.15 | yes |
| **BERTurk (quantized)** | **INT8 (dynamic)** | **0.810** | **177.4** | **17.55** | **25.11** | **yes** |

Quantization made BERTurk 2.4× smaller and 1.8× faster (p95: 44 → 25 ms) for −0.003 macro-F1.
The full table is in [`results/results_table.md`](results/results_table.md).

---

## Model ladder and techniques

| Area | Component | File |
|---|---|---|
| B · Classical NLP | Regex normalization of noisy text (leetspeak, repeated letters, separators, Turkish I/İ) | `src/text_cleaning.py` |
| B · Classical NLP | Rule baseline: lexicon + minimum edit distance (Kaggle list filtered with training data + words learned from data) | `src/build_lexicon.py`, `src/lexicon_matcher.py` |
| B · Classical NLP | tf-idf (word 1–2 grams + char 2–5 grams) + logistic regression | `src/train_tfidf.py` |
| B · Classical NLP | Our own word2vec (skip-gram) trained on the training tweets | `src/train_word2vec.py` |
| A · Neural networks | Feedforward network (dropout) trained with SGD + backpropagation on word2vec vectors | `src/neural_model.py`, `src/train_ffnn.py` |
| C · Transformers | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned for classification, WordPiece tokenizer | `src/train_bert.py` |
| D · Efficient models | Dynamic INT8 quantization + benchmarking of size, latency and macro-F1 | `src/quantize_bert.py`, `src/evaluate_all.py` |

Data split: the official training set is split 90 % / 10 % (stratified, seed 42) into training and
validation; all settings (lexicon choice, C, epochs) were chosen on the validation set. The official
test set was used only once, for the final table.

---

## Reproduce the results

Run from the project folder, in this order:

```bash
python src/prepare_data.py        # download OffensEval-TR, create train / validation / test CSVs
python src/build_lexicon.py       # build the three lexicons + safe-word list (training data only)
python src/evaluate_lexicons.py   # pick the best lexicon on the validation set
python src/train_tfidf.py         # tf-idf + logistic regression
python src/train_word2vec.py      # our own word2vec
python src/train_ffnn.py          # feedforward network (SGD)
python src/train_bert.py          # fine-tune BERTurk (GPU strongly recommended)
python src/quantize_bert.py       # INT8 quantization
python src/evaluate_all.py        # final test-set table -> results/
```

For GPU training on an NVIDIA card, replace the CPU build of PyTorch with the CUDA build first:
```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```
BERTurk was fine-tuned on an RTX 3050 Ti Laptop GPU (4 GB) in about 9 minutes.

---

## Project structure

```
chat-moderation/
├── data/
│   ├── kaggle_swear_words.csv   # Kaggle "Turkish slang and swear words" list (Apache 2.0)
│   ├── lexicon_*.txt            # lexicons built by build_lexicon.py
│   ├── lexicon_selected.txt     # lexicon used by the app
│   └── safe_words.txt           # common clean words, skipped by the matcher
├── models/                      # downloaded from the Hugging Face Hub (not in Git)
├── results/                     # results table (CSV + Markdown)
├── src/
│   ├── app.py                   # Gradio app
│   ├── classifiers.py           # one interface for all five models
│   ├── download_models.py       # fetches the models from the Hugging Face Hub
│   ├── text_cleaning.py, lexicon_matcher.py, neural_model.py, bert_model.py
│   └── prepare_data.py, build_lexicon.py, train_*.py, quantize_bert.py, evaluate_*.py
└── requirements.txt
```

---

## Data sources and credits

- **OffensEval-TR 2020** – Çağrı Çöltekin. 2020. *A Corpus of Turkish Offensive Language on Social Media.*
  In Proceedings of LREC 2020. Loaded from the Hugging Face Hub (`coltekin/offenseval2020_tr`).
- **Turkish slang and swear words** – Kaggle dataset by *aiwithcagri*, Apache License 2.0
  (https://www.kaggle.com/datasets/aiwithcagri/turkish-slang-and-swear-trke-kfrler).
- **BERTurk** – `dbmdz/bert-base-turkish-cased` by the MDZ Digital Library team (Bavarian State Library).

## Limitations

- Rule-based matching is fooled by creative spellings (e.g. `ap.taaal` is not caught), and removing
  Turkish characters creates collisions (`sıkıcı` → `sikici`).
- Data-driven lexicons can learn bias: `türbanlı` appeared mostly in offensive tweets and had to be
  removed manually.
- All models over-fit after a few epochs; the best epoch was chosen on the validation set.

## AI usage

AI assistants were used during development; see the AI-usage slide in the presentation.
