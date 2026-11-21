"""
Step 3 - Build the offensive word lists (lexicons) for the rule-based baseline.

Creates three lexicons that are compared later on the validation set:
  A        : the Kaggle "Turkish slang and swear words" list (Apache 2.0),
             filtered with our training data and a small manual exclude list
  B        : words that are strongly linked to offensive tweets in our training data
  combined : A + B

It also creates a "safe words" list: common words (written WITH Turkish
characters) that appear mostly in clean tweets, e.g. "sıkıcı" or "golü".
Without Turkish characters these look like swear words ("sikici", "gotlu"),
so the matcher skips them.

Only data/train.csv is used here. The validation and test sets are not touched.
All words are stored in normalized form: clean_full + to_ascii.
"""
from collections import Counter
from pathlib import Path

import pandas as pd

from text_cleaning import clean_full, to_ascii, tokenize

DATA_DIR = Path("data")
KAGGLE_FILE = DATA_DIR / "kaggle_swear_words.csv"

# Lexicon A: a Kaggle word is dropped if it appears in at least MIN_COUNT
# training tweets and less than KEEP_RATIO of those tweets are offensive.
MIN_COUNT = 5
KEEP_RATIO = 0.5

# Lexicon B: a training word is added if it appears in at least
# NEW_WORD_MIN_COUNT offensive tweets and at least NEW_WORD_RATIO of all
# tweets that contain it are offensive.
NEW_WORD_MIN_COUNT = 10
NEW_WORD_RATIO = 0.8

# Safe words: appear in at least SAFE_MIN_COUNT training tweets and less than
# SAFE_MAX_RATIO of those tweets are offensive.
SAFE_MIN_COUNT = 5
SAFE_MAX_RATIO = 0.3

# Words that are harmless on their own, too generic, or wrongly included in the
# Kaggle list (names, cities, abbreviations, parts of longer phrases).
MANUAL_EXCLUDE_RAW = [
    "ab", "ag", "allah", "allahsız", "ana", "anam", "anan", "anana", "anandan", "anas",
    "anay", "anneni", "annenin", "annesiz", "babani", "babanı", "babanın", "abuzer",
    "attırdığım", "azdım", "azdır", "biting", "cif", "cikar", "cim", "diktim", "dinsiz",
    "dönek", "düdük", "ecdadını", "ecdadini", "emenike", "emi", "esc", "ferre",
    "feriştah", "giresunlu", "hoşafı", "imansz", "kaka", "kaypak", "kayyum", "keleş",
    "kelle", "koyarm", "koyayım", "koyiim", "koyiiym", "koyim", "koyum", "koyyim", "krar",
    "kısaltma", "küfür", "lahmi", "meme", "oe", "okiyen", "oğlan", "ökkeş", "pipi",
    "rahminde", "revizyonist", "saksofon", "samsunlu", "sekis", "sevişelim", "sgk",
    "sie", "slaleni", "sulaleni", "sülaleni", "sülalenizi", "tiyniyat", "toplarm",
    "topsun", "uşak", "vahap", "verdiimin", "wtf", "wtff",
    "türbanlı", "zekalı", "yoksunu", "görünümlü",
]


def normalize(text: str) -> str:
    return to_ascii(clean_full(text))


MANUAL_EXCLUDE = {normalize(word) for word in MANUAL_EXCLUDE_RAW}


def save_lexicon(words: set[str], path: Path) -> None:
    path.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")


# 1) Count in how many clean / offensive training tweets each word appears.
train = pd.read_csv(DATA_DIR / "train.csv")
offensive_counts = Counter()
clean_counts = Counter()
for text, label in zip(train["text"], train["label"]):
    words = set(tokenize(normalize(text)))  # set(): count a word once per tweet
    if label == 1:
        offensive_counts.update(words)
    else:
        clean_counts.update(words)


def word_stats(word: str) -> tuple[int, float]:
    """Number of training tweets containing the word, and the offensive share."""
    total = offensive_counts[word] + clean_counts[word]
    ratio = offensive_counts[word] / total if total else 0.0
    return total, ratio


# 2) Lexicon A: Kaggle list, single words only, filtered.
kaggle = pd.read_csv(KAGGLE_FILE)
kaggle_words = set()
for entry in kaggle["kufur"].dropna():
    words = tokenize(normalize(entry))
    if len(words) == 1:
        kaggle_words.add(words[0])

lexicon_a = set()
dropped_manual = []
dropped_by_data = []
for word in sorted(kaggle_words):
    total, ratio = word_stats(word)
    if word in MANUAL_EXCLUDE:
        dropped_manual.append(word)
    elif total >= MIN_COUNT and ratio < KEEP_RATIO:
        dropped_by_data.append((word, total, ratio))
    else:
        lexicon_a.add(word)

# 3) Lexicon B: words learned from the training data.
lexicon_b = set()
for word, offensive in offensive_counts.items():
    total, ratio = word_stats(word)
    if (
        offensive >= NEW_WORD_MIN_COUNT
        and ratio >= NEW_WORD_RATIO
        and len(word) >= 2
        and not word.isdigit()
        and word not in MANUAL_EXCLUDE
    ):
        lexicon_b.add(word)

lexicon_combined = lexicon_a | lexicon_b

# 4) Safe words: counted on clean_full text, i.e. WITH Turkish characters.
safe_offensive = Counter()
safe_total = Counter()
for text, label in zip(train["text"], train["label"]):
    words = set(tokenize(clean_full(text)))
    safe_total.update(words)
    if label == 1:
        safe_offensive.update(words)
safe_words = {
    w for w, total in safe_total.items()
    if total >= SAFE_MIN_COUNT and safe_offensive[w] / total < SAFE_MAX_RATIO
}

save_lexicon(lexicon_a, DATA_DIR / "lexicon_a_kaggle.txt")
save_lexicon(lexicon_b, DATA_DIR / "lexicon_b_data.txt")
save_lexicon(lexicon_combined, DATA_DIR / "lexicon_combined.txt")
save_lexicon(safe_words, DATA_DIR / "safe_words.txt")

# 5) Report
print(f"Kaggle single words (after normalization): {len(kaggle_words)}")
print(f"  removed by manual list : {len(dropped_manual)}")
print(f"  removed by data filter : {len(dropped_by_data)}")
print(f"Lexicon A (Kaggle)  : {len(lexicon_a)} words")
print(f"Lexicon B (data)    : {len(lexicon_b)} words")
print(f"Lexicon A+B         : {len(lexicon_combined)} words")
print(f"Safe words          : {len(safe_words)} words")

print("\n--- Kaggle words removed by the data filter (tweets, offensive share) ---")
for word, total, ratio in sorted(dropped_by_data, key=lambda x: -x[1]):
    print(f"  {word:<15} {total:>5}  {ratio:.0%}")

print("\n--- Lexicon B words (tweets, offensive share), please review ---")
for word in sorted(lexicon_b, key=lambda w: -word_stats(w)[0]):
    total, ratio = word_stats(word)
    marker = "" if word in lexicon_a else "  (new, not in Kaggle list)"
    print(f"  {word:<15} {total:>5}  {ratio:.0%}{marker}")