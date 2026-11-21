"""
Step 6a - Train our own word2vec embeddings on the training tweets (technique area B).

word2vec learns a vector of numbers for every word, so that words used in
similar contexts get similar vectors ("salak" ends up close to "aptal").
We train it only on data/train.csv, with the same clean_full cleaning
as the other classical models.

Saves models/word2vec.model
"""
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

from text_cleaning import clean_full, tokenize

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "word2vec.model"

VECTOR_SIZE = 100   # length of each word vector
WINDOW = 5          # how many neighbouring words count as "context"
MIN_COUNT = 2       # ignore words that appear only once
EPOCHS = 15         # how many times we go over all tweets

MODELS_DIR.mkdir(exist_ok=True)
train = pd.read_csv(DATA_DIR / "train.csv")
sentences = [tokenize(clean_full(text)) for text in train["text"]]

print(f"Training word2vec on {len(sentences)} tweets ...")
model = Word2Vec(
    sentences,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=1,          # skip-gram: predict the neighbours of each word
    epochs=EPOCHS,
    seed=42,
    workers=1,     # one thread, so that the result is reproducible
)
model.save(str(MODEL_FILE))
print(f"Vocabulary size: {len(model.wv)} words")
print(f"Saved to {MODEL_FILE}")

print("\nNearest neighbours (words with the most similar vectors):")
for word in ["salak", "amk", "güzel", "maç", "hoca", "teşekkürler"]:
    if word in model.wv:
        neighbours = ", ".join(w for w, _ in model.wv.most_similar(word, topn=6))
        print(f"  {word:<12} -> {neighbours}")