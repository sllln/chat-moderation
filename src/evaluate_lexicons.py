"""
Step 4 - Compare the three lexicons on the VALIDATION set and pick the best one.

The best lexicon (highest macro-F1) is copied to data/lexicon_selected.txt
and is the one used by the rule-based baseline from now on.
"""
import shutil
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from lexicon_matcher import LexiconMatcher

DATA_DIR = Path("data")
LEXICONS = {
    "A (Kaggle)": DATA_DIR / "lexicon_a_kaggle.txt",
    "B (data)": DATA_DIR / "lexicon_b_data.txt",
    "A+B (combined)": DATA_DIR / "lexicon_combined.txt",
}
SAFE_WORDS_FILE = DATA_DIR / "safe_words.txt"
SELECTED_FILE = DATA_DIR / "lexicon_selected.txt"

validation = pd.read_csv(DATA_DIR / "validation.csv")
texts = validation["text"].tolist()
labels = validation["label"].tolist()

print(f"Validation set: {len(texts)} tweets\n")
print(f"{'lexicon':<16} {'words':>6} {'macro-F1':>9} {'precision':>10} {'recall':>7} {'time':>7}")

results = {}
for name, path in LEXICONS.items():
    matcher = LexiconMatcher.from_files(path, SAFE_WORDS_FILE)
    start = time.perf_counter()
    predictions = [matcher.predict(text) for text in texts]
    seconds = time.perf_counter() - start

    macro_f1 = f1_score(labels, predictions, average="macro")
    precision = precision_score(labels, predictions)  # of the flagged tweets, how many were really offensive
    recall = recall_score(labels, predictions)        # of the offensive tweets, how many were flagged
    results[name] = (macro_f1, matcher, predictions)
    print(f"{name:<16} {len(matcher.words):>6} {macro_f1:>9.3f} {precision:>10.3f} {recall:>7.3f} {seconds:>6.1f}s")

best_name = max(results, key=lambda n: results[n][0])
shutil.copy(LEXICONS[best_name], SELECTED_FILE)
print(f"\nBest lexicon: {best_name} -> copied to {SELECTED_FILE}")

# Show some mistakes of the best lexicon, so we can learn from them.
_, matcher, predictions = results[best_name]
false_alarms = [t for t, y, p in zip(texts, labels, predictions) if y == 0 and p == 1]
missed = [t for t, y, p in zip(texts, labels, predictions) if y == 1 and p == 0]

print(f"\n--- 10 false alarms (clean tweets flagged as offensive), out of {len(false_alarms)} ---")
for text in false_alarms[:10]:
    print(f"  {matcher.find_matches(text)}  <-  {text[:100]}")

print(f"\n--- 10 missed offensive tweets, out of {len(missed)} ---")
for text in missed[:10]:
    print(f"  {text[:120]}")