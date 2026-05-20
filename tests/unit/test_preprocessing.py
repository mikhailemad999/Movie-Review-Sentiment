"""Unit tests for the cleaning pipeline and tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from sentiment.data.preprocessing import (
    CleaningConfig, SentimentTokenizer,
    clean, normalise_contractions, remove_html, remove_stopwords,
)
from sentiment.exceptions import TokenizerNotFittedError

CORPUS = ["This movie was absolutely amazing and wonderful.",
          "Terrible film, complete waste of time.",
          "Pretty average — some good moments."] * 20


class TestCleaningFunctions:
    def test_remove_html(self) -> None:
        assert "<b>" not in remove_html("<b>Great</b> film!")

    def test_normalise_contraction(self) -> None:
        assert "will not" in normalise_contractions("i won't go")

    def test_negation_preserved(self) -> None:
        assert "not" in remove_stopwords("i do not like this film")

    def test_clean_pipeline(self) -> None:
        result = clean("<p>This is NOT a great film!</p>")
        assert "<" not in result
        assert "not" in result

    def test_clean_empty(self) -> None:
        assert clean("   ") == ""


class TestSentimentTokenizer:
    def test_fit_transform_shape(self) -> None:
        tok = SentimentTokenizer(vocab_size=200, max_len=30)
        X   = tok.fit_transform(CORPUS)
        assert X.shape == (len(CORPUS), 30)
        assert X.dtype == np.int32

    def test_not_fitted_raises(self) -> None:
        with pytest.raises(TokenizerNotFittedError):
            SentimentTokenizer().transform(["hello world"])

    def test_save_load_roundtrip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        tok = SentimentTokenizer(vocab_size=200, max_len=30)
        tok.fit(CORPUS)
        path = tmp_path / "tokenizer.pkl"
        tok.save(path)
        tok2 = SentimentTokenizer.load(path)
        np.testing.assert_array_equal(
            tok.transform(["A great movie!"]),
            tok2.transform(["A great movie!"]),
        )

    def test_effective_vocab_capped(self) -> None:
        tok = SentimentTokenizer(vocab_size=5, max_len=10)
        tok.fit(CORPUS)
        assert tok.effective_vocab_size <= 6
