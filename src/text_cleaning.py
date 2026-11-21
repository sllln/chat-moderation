"""
Text normalization rules for noisy Turkish chat messages (technique area B).

Two cleaning levels are provided:
  - clean_light(text): for BERTurk. Fixes the noise but keeps letter case,
    Turkish characters and punctuation, because BERTurk was trained on
    normal, well-written Turkish.
  - clean_full(text):  for the classical models (lexicon matcher, tf-idf,
    word2vec). Applies clean_light, then lowercases (Turkish-aware) and
    removes everything that is not a letter, a digit or a space.

to_ascii(text) additionally maps Turkish characters to their ASCII look-alikes
(ş->s, ı->i, ...). It is only used by the lexicon matcher, because people often
type "aptalsin" instead of "aptalsın".
"""
import re

# --- Patterns -----------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")

# Single letters separated by . - _ * (e.g. "a.p.t.a.l", "s-a-l-a-k", "a.q.")
SEPARATED_LETTERS_PATTERN = re.compile(r"\b(?:[^\W\d_][.\-_*]){1,}[^\W\d_]\b[.\-_*]?")

# The same character three or more times in a row (e.g. "çoooook", "!!!!")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

# Characters that are used instead of letters ("leetspeak")
LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"}

# Python's lower() turns "I" into "i", but in Turkish it must become "ı",
# and "İ" must become "i". We handle these two letters ourselves.
TURKISH_UPPER_TO_LOWER = str.maketrans({"I": "ı", "İ": "i"})

TURKISH_TO_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîû", "cgiosuCGIOSUaiu")


# --- Single steps -------------------------------------------------------------

def turkish_lower(text: str) -> str:
    """Lowercase a string using Turkish rules (I -> ı, İ -> i)."""
    return text.translate(TURKISH_UPPER_TO_LOWER).lower()


def to_ascii(text: str) -> str:
    """Replace Turkish characters with their ASCII look-alikes."""
    return text.translate(TURKISH_TO_ASCII)


def decode_leetspeak(text: str) -> str:
    """Turn digits/symbols that stand for letters back into letters.

    A character is only replaced when it touches a letter and does not
    touch a digit. So "s3n" -> "sen", but numbers such as "2024", "3-0"
    or "10.30da" stay unchanged.
    """
    words = []
    for word in text.split(" "):
        chars = list(word)
        for i, char in enumerate(chars):
            if char not in LEET_MAP:
                continue
            left = chars[i - 1] if i > 0 else ""
            right = chars[i + 1] if i + 1 < len(chars) else ""
            touches_letter = left.isalpha() or right.isalpha()
            touches_digit = left.isdigit() or right.isdigit()
            if touches_letter and not touches_digit:
                chars[i] = LEET_MAP[char]
        words.append("".join(chars))
    return " ".join(words)


def join_separated_letters(text: str) -> str:
    """'a.p.t.a.l' -> 'aptal'"""
    return SEPARATED_LETTERS_PATTERN.sub(lambda m: re.sub(r"[.\-_*]", "", m.group(0)), text)


def squeeze_repeats(text: str) -> str:
    """Three or more identical characters become one: 'çoooook' -> 'çok'.

    Two identical characters are kept, because Turkish has real double
    letters ("saat", "hakkında").
    """
    return REPEATED_CHAR_PATTERN.sub(r"\1", text)


# --- Full pipelines -------------------------------------------------------------

def clean_light(text: str) -> str:
    """Light cleaning for BERTurk (keeps case, Turkish letters, punctuation)."""
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = join_separated_letters(text)
    text = decode_leetspeak(text)
    text = squeeze_repeats(text)
    return text


def clean_full(text: str) -> str:
    """Full cleaning for the classical models."""
    text = clean_light(text)
    text = turkish_lower(text)
    text = re.sub(r"[^\w\s]|_", " ", text)   # drop punctuation and emojis
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split an already cleaned text into words."""
    return text.split()