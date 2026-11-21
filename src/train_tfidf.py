"""
Step 5 - Classical baseline: tf-idf features + logistic regression (technique area B).

  - Text is cleaned with clean_full (inside the vectorizers).
  - Two kinds of tf-idf features are combined:
      * word n-grams (single words and word pairs),
      * character n-grams (2-5 letters), which handle Turkish suffixes and
        typos: "aptal", "aptallar", "aptalsın" share many character pieces.
  - Logistic regression learns one weight per feature.
  - The regularization strength C is chosen on the VALIDATION set.

The best model is saved to models/tfidf_logreg.joblib
"""
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from text_cleaning import clean_full

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "tfidf_logreg.joblib"
C_VALUES = [0.5, 1, 2, 4, 8]   # smaller C = simpler model, larger C = fits the training data more closely


def build_pipeline(c: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_full, analyzer="word",
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=clean_full, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
    return Pipeline([("features", features), ("classifier", classifier)])


MODELS_DIR.mkdir(exist_ok=True)
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")

print(f"{'C':>5} {'validation macro-F1':>20} {'training time':>14}")
best_f1, best_c, best_model = -1.0, None, None
for c in C_VALUES:
    start = time.perf_counter()
    model = build_pipeline(c).fit(train["text"], train["label"])
    seconds = time.perf_counter() - start
    f1 = f1_score(validation["label"], model.predict(validation["text"]), average="macro")
    print(f"{c:>5} {f1:>20.3f} {seconds:>13.1f}s")
    if f1 > best_f1:
        best_f1, best_c, best_model = f1, c, model

joblib.dump(best_model, MODEL_FILE)
size_mb = MODEL_FILE.stat().st_size / 1024 / 1024
print(f"\nBest C = {best_c} (validation macro-F1 = {best_f1:.3f})")
print(f"Saved to {MODEL_FILE} ({size_mb:.1f} MB)")

# Which words push a message most strongly towards "offensive"?
names = best_model.named_steps["features"].get_feature_names_out()
weights = best_model.named_steps["classifier"].coef_[0]
word_features = [(w, n.removeprefix("word__")) for n, w in zip(names, weights) if n.startswith("word__")]
word_features.sort(reverse=True)
print("\nTop 20 word features for 'offensive':")
print(", ".join(name for _, name in word_features[:20]))
print("\nTop 20 word features for 'clean':")
print(", ".join(name for _, name in word_features[-20:]))