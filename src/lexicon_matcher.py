"""
Rule-based baseline (technique area B): lexicon matcher with minimum edit distance.

A message is labeled offensive if at least one of its words matches a word
in the offensive lexicon:
  1) exactly,
  2) as a prefix, because Turkish adds suffixes ("aptal" -> "aptallar"),
  3) with a small number of typos, measured by the minimum edit distance.

Both the message and the lexicon are normalized with clean_full + to_ascii.
Words from the "safe words" list (common clean words written with Turkish
characters, e.g. "sıkıcı") are skipped before matching.
"""
from functools import lru_cache
from pathlib import Path

from text_cleaning import clean_full, to_ascii, tokenize

# Lexicon words must be at least this long to be used as a prefix
# (shorter ones such as "got" or "bok" would match too many normal words).
MIN_PREFIX_LENGTH = 5


def edit_distance(a: str, b: str) -> int:
    """Minimum number of insertions, deletions and substitutions that turn a into b.

    Classic dynamic programming (Levenshtein). We only keep the previous row
    of the table in memory.
    """
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(
                previous[j] + 1,         # delete char_a
                current[j - 1] + 1,      # insert char_b
                previous[j - 1] + cost,  # substitute (or keep, if equal)
            ))
        previous = current
    return previous[-1]


def allowed_distance(word_length: int) -> int:
    """How many typos we accept, depending on the length of the lexicon word."""
    if word_length <= 5:
        return 0   # short words: exact match only ("kahpe" must not match "kahve")
    if word_length <= 8:
        return 1
    return 2


class LexiconMatcher:
    def __init__(self, words, safe_words=()):
        self.words = set(words)
        self.safe_words = set(safe_words)
        self.prefix_words = [w for w in self.words if len(w) >= MIN_PREFIX_LENGTH]
        self.fuzzy_words = [w for w in self.words if allowed_distance(len(w)) > 0]
        # Remember the answer for every word we have already checked.
        self._match_token = lru_cache(maxsize=200_000)(self._match_token_uncached)

    @classmethod
    def from_files(cls, lexicon_path, safe_words_path=None):
        words = Path(lexicon_path).read_text(encoding="utf-8").split()
        safe = Path(safe_words_path).read_text(encoding="utf-8").split() if safe_words_path else []
        return cls(words, safe)

    def _match_token_uncached(self, token: str):
        """Return the lexicon word that matches this token, or None."""
        if token in self.words:
            return token
        for word in self.prefix_words:
            if token.startswith(word):
                return word
        for word in self.fuzzy_words:
            max_distance = allowed_distance(len(word))
            if abs(len(word) - len(token)) > max_distance:
                continue
            if edit_distance(token, word) <= max_distance:
                return word
        return None

    def find_matches(self, text: str) -> list[tuple[str, str]]:
        """All (message word, lexicon word) pairs that matched."""
        matches = []
        for word in tokenize(clean_full(text)):
            if word in self.safe_words:
                continue
            token = to_ascii(word)
            match = self._match_token(token)
            if match is not None:
                matches.append((token, match))
        return matches

    def predict(self, text: str) -> int:
        """1 = offensive, 0 = clean"""
        return 1 if self.find_matches(text) else 0