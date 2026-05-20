"""
preprocessing.py — Text cleaning pipeline and tokenizer wrapper.

Design goals:
  * Deterministic: same input -> same output, no hidden state.
  * Tested: each transform is a pure function.
  * Sentiment-safe: negation words are never stripped.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import nltk
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from sentiment.exceptions import TokenizerNotFittedError

log = logging.getLogger(__name__)

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

_NEGATION = frozenset({
    "not", "no", "nor", "never", "neither", "nobody",
    "nothing", "nowhere", "hardly", "barely", "scarcely", "without"
})
_STOP_WORDS = frozenset(stopwords.words("english")) - _NEGATION


# ── Pure cleaning functions ───────────────────────────────────────────────────

def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+", " ", text)

def keep_alpha(text: str) -> str:
    return re.sub(r"[^a-z\s]", " ", text)

def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalise_contractions(text: str) -> str:
    contractions = {
        r"won't": "will not", r"can't": "cannot",   r"n't":   " not",
        r"i'm":   "i am",     r"i've":  "i have",   r"i'll":  "i will",
        r"i'd":   "i would",  r"it's":  "it is",    r"that's": "that is",
        r"there's": "there is", r"they're": "they are",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text

def remove_stopwords(text: str, stop_words: frozenset[str] = _STOP_WORDS) -> str:
    return " ".join(w for w in text.split() if w not in stop_words)


# ── Cleaning pipeline ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CleaningConfig:
    lowercase:           bool = True
    expand_contractions: bool = True
    strip_html:          bool = True
    strip_urls:          bool = True
    alpha_only:          bool = True
    strip_stops:         bool = True


def clean(text: str, cfg: CleaningConfig = CleaningConfig()) -> str:
    if cfg.lowercase:          text = text.lower()
    if cfg.expand_contractions: text = normalise_contractions(text)
    if cfg.strip_html:         text = remove_html(text)
    if cfg.strip_urls:         text = remove_urls(text)
    if cfg.alpha_only:         text = keep_alpha(text)
    if cfg.strip_stops:        text = remove_stopwords(text)
    return collapse_spaces(text)


# ── Tokenizer wrapper ─────────────────────────────────────────────────────────

class SentimentTokenizer:
    """
    Thin wrapper around Keras Tokenizer.
    Provides fit/transform API consistent with scikit-learn conventions.
    """

    def __init__(
        self,
        vocab_size:   int           = 10_000,
        max_len:      int           = 200,
        oov_token:    str           = "<OOV>",
        cleaning_cfg: CleaningConfig = CleaningConfig(),
    ) -> None:
        self.vocab_size   = vocab_size
        self.max_len      = max_len
        self.oov_token    = oov_token
        self.cleaning_cfg = cleaning_cfg
        self._tokenizer   = Tokenizer(num_words=vocab_size, oov_token=oov_token)
        self._fitted      = False

    def fit(self, texts: Sequence[str]) -> "SentimentTokenizer":
        cleaned = [clean(t, self.cleaning_cfg) for t in texts]
        self._tokenizer.fit_on_texts(cleaned)
        self._fitted = True
        log.info("Tokenizer fitted on %d texts — vocab: %d", len(texts), len(self._tokenizer.word_index))
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise TokenizerNotFittedError("Call .fit() before .transform()")
        cleaned = [clean(t, self.cleaning_cfg) for t in texts]
        seqs    = self._tokenizer.texts_to_sequences(cleaned)
        return pad_sequences(seqs, maxlen=self.max_len, padding="post", truncating="post")

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def word_index(self) -> dict[str, int]:
        if not self._fitted:
            raise TokenizerNotFittedError
        return self._tokenizer.word_index  # type: ignore[return-value]

    @property
    def effective_vocab_size(self) -> int:
        return min(self.vocab_size, len(self.word_index) + 1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Tokenizer saved -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SentimentTokenizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected SentimentTokenizer, got {type(obj)}")
        return obj
