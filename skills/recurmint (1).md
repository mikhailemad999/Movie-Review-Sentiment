# 🎬 GRU Sentiment Analysis — Senior-Level Production Project

> **Enterprise-grade** movie review sentiment classification using a Bidirectional GRU with
> self-attention. Features a typed config system, MLflow experiment tracking, FastAPI REST
> service, Streamlit dashboard, full test suite, Docker deployment, and CI/CD pipeline.

---

## 📁 Project Layout

```
sentiment_gru/
├── src/
│   └── sentiment/
│       ├── __init__.py
│       ├── config.py            # Pydantic-based typed config
│       ├── exceptions.py        # Custom exception hierarchy
│       ├── registry.py          # Model registry (versioning)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py       # tf.data pipeline
│       │   └── preprocessing.py # Cleaner + Tokenizer wrapper
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py          # Abstract model interface
│       │   ├── gru.py           # Bidirectional GRU + Attention
│       │   └── factory.py       # Model factory pattern
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py       # Training orchestrator
│       │   ├── callbacks.py     # Custom Keras callbacks
│       │   └── metrics.py       # Extended metrics
│       └── serving/
│           ├── __init__.py
│           ├── predictor.py     # Inference engine
│           ├── api.py           # FastAPI REST service
│           └── schemas.py       # Pydantic request/response models
├── app/
│   └── streamlit_app.py         # Production Streamlit dashboard
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   ├── test_model.py
│   │   └── test_schemas.py
│   └── integration/
│       ├── test_trainer.py
│       └── test_api.py
├── scripts/
│   ├── train.py                 # CLI entry-point (Typer)
│   ├── evaluate.py              # Evaluation report CLI
│   └── export_model.py          # TF-SavedModel / ONNX export
├── configs/
│   ├── default.yaml
│   ├── experiment_gru_small.yaml
│   └── experiment_gru_large.yaml
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.app
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml               # Lint + test on every PR
│       └── cd.yml               # Build & push Docker on main merge
├── pyproject.toml               # PEP 517/518 build config
├── setup.cfg                    # Tool config (mypy, flake8, pytest)
├── Makefile                     # Dev shortcuts
└── README.md
```

---

## pyproject.toml

```toml
[build-system]
requires      = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name        = "sentiment-gru"
version     = "1.0.0"
description = "Production GRU sentiment analysis — IMDB"
requires-python = ">=3.10"
license     = { text = "MIT" }
authors     = [{ name = "Your Name", email = "you@example.com" }]

dependencies = [
    "tensorflow>=2.14",
    "numpy>=1.26",
    "pandas>=2.1",
    "scikit-learn>=1.3",
    "pydantic>=2.4",
    "pydantic-settings>=2.0",
    "typer[all]>=0.9",
    "rich>=13.6",
    "mlflow>=2.8",
    "fastapi>=0.104",
    "uvicorn[standard]>=0.24",
    "httpx>=0.25",
    "streamlit>=1.28",
    "plotly>=5.18",
    "nltk>=3.8",
    "PyYAML>=6.0",
    "prometheus-client>=0.18",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.1",
    "httpx>=0.25",
    "mypy>=1.6",
    "ruff>=0.1",
    "pre-commit>=3.5",
    "faker>=20.0",
]

[project.scripts]
sentiment-train    = "scripts.train:app"
sentiment-evaluate = "scripts.evaluate:app"
sentiment-serve    = "sentiment.serving.api:start"

[tool.setuptools.packages.find]
where = ["src"]
```

---

## setup.cfg — Tool Configuration

```ini
[mypy]
python_version      = 3.10
strict              = true
ignore_missing_imports = true
plugins             = pydantic.mypy

[flake8]
max-line-length = 100
extend-ignore   = E203, W503

[tool:pytest]
testpaths         = tests
asyncio_mode      = auto
addopts           = -v --cov=src/sentiment --cov-report=term-missing --cov-fail-under=80

[coverage:run]
source  = src/sentiment
omit    = tests/*

[ruff]
line-length    = 100
select         = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore         = ["ANN101"]
target-version = "py310"
```

---

## configs/default.yaml

```yaml
data:
  vocab_size:       10000
  max_sequence_len: 200
  test_split:       0.2
  val_split:        0.1
  random_seed:      42

model:
  name:          "gru_sentiment_v1"
  embedding_dim: 128
  gru_units:     [128, 64]
  attention:     true
  dense_units:   [128, 64]
  dropout_rate:  0.3
  spatial_dropout: 0.2
  l2_reg:        0.0001

training:
  batch_size:      64
  epochs:          15
  learning_rate:   0.001
  lr_warmup_steps: 500
  weight_decay:    0.01
  label_smoothing: 0.1
  early_stopping_patience: 4
  reduce_lr_patience:      2
  reduce_lr_factor:        0.5
  min_lr:                  1.0e-7
  mixed_precision:         false

logging:
  level:        "INFO"
  mlflow_uri:   "http://localhost:5000"
  experiment:   "gru-sentiment"
  log_model:    true

paths:
  artifacts:   "artifacts/"
  model_dir:   "artifacts/models/"
  logs_dir:    "artifacts/logs/"
  reports_dir: "artifacts/reports/"
```

---

## src/sentiment/config.py

```python
"""
config.py — Typed, validated configuration with Pydantic v2.

All settings flow from YAML files or environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    vocab_size:       Annotated[int,   Field(gt=0)]           = 10_000
    max_sequence_len: Annotated[int,   Field(gt=0)]           = 200
    test_split:       Annotated[float, Field(gt=0, lt=1)]     = 0.2
    val_split:        Annotated[float, Field(gt=0, lt=1)]     = 0.1
    random_seed:      int                                      = 42


class ModelConfig(BaseModel):
    name:            str       = "gru_sentiment_v1"
    embedding_dim:   int       = 128
    gru_units:       list[int] = [128, 64]
    attention:       bool      = True
    dense_units:     list[int] = [128, 64]
    dropout_rate:    float     = 0.3
    spatial_dropout: float     = 0.2
    l2_reg:          float     = 1e-4

    @model_validator(mode="after")
    def validate_units(self) -> "ModelConfig":
        if not self.gru_units:
            raise ValueError("gru_units must not be empty")
        return self


class TrainingConfig(BaseModel):
    batch_size:               int   = 64
    epochs:                   int   = 15
    learning_rate:            float = 1e-3
    lr_warmup_steps:          int   = 500
    weight_decay:             float = 0.01
    label_smoothing:          float = 0.1
    early_stopping_patience:  int   = 4
    reduce_lr_patience:       int   = 2
    reduce_lr_factor:         float = 0.5
    min_lr:                   float = 1e-7
    mixed_precision:          bool  = False


class LoggingConfig(BaseModel):
    level:      str  = "INFO"
    mlflow_uri: str  = "http://localhost:5000"
    experiment: str  = "gru-sentiment"
    log_model:  bool = True


class PathsConfig(BaseModel):
    artifacts:   Path = Path("artifacts/")
    model_dir:   Path = Path("artifacts/models/")
    logs_dir:    Path = Path("artifacts/logs/")
    reports_dir: Path = Path("artifacts/reports/")

    @model_validator(mode="after")
    def create_dirs(self) -> "PathsConfig":
        for p in (self.artifacts, self.model_dir, self.logs_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


class AppConfig(BaseModel):
    """Root configuration object — load via AppConfig.from_yaml()."""

    data:     DataConfig     = DataConfig()
    model:    ModelConfig    = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    logging:  LoggingConfig  = LoggingConfig()
    paths:    PathsConfig    = PathsConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg_path = os.environ.get("SENTIMENT_CONFIG", "configs/default.yaml")
        return cls.from_yaml(cfg_path)


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env")

    host:            str  = "0.0.0.0"
    port:            int  = 8000
    workers:         int  = 1
    log_level:       str  = "info"
    model_path:      Path = Path("artifacts/models/best_model")
    tokenizer_path:  Path = Path("artifacts/models/tokenizer.pkl")
    max_batch:       int  = 64
    request_timeout: float = 30.0
    enable_metrics:  bool = True
```

---

## src/sentiment/exceptions.py

```python
"""
exceptions.py — Custom exception hierarchy.
"""

from __future__ import annotations


class SentimentBaseError(Exception):
    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.context = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class DataError(SentimentBaseError):
    """Dataset loading / preprocessing failures."""

class TokenizerNotFittedError(DataError):
    """predict() called before fit()."""

class ModelError(SentimentBaseError):
    """Model construction or forward-pass failures."""

class ModelNotFoundError(ModelError):
    """Saved model artifact cannot be located."""

class ModelLoadError(ModelError):
    """Deserialization of a saved model failed."""

class TrainingError(SentimentBaseError):
    """Training-loop failures."""

class CheckpointError(TrainingError):
    """Saving / loading a checkpoint failed."""

class ServingError(SentimentBaseError):
    """Inference-pipeline failures."""

class InvalidInputError(ServingError):
    """API received malformed or out-of-range input."""
```

---

## src/sentiment/data/preprocessing.py

```python
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
```

---

## src/sentiment/data/dataset.py

```python
"""
dataset.py — High-performance tf.data input pipeline.

Features:
  * Parallel preprocessing with tf.data
  * Caching and prefetching
  * Stratified train/val split
  * Class-weight computation for imbalanced datasets
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.datasets import imdb

from sentiment.config import DataConfig
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.exceptions import DataError

log     = logging.getLogger(__name__)
AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class Split:
    dataset:   tf.data.Dataset
    size:      int
    pos_ratio: float


@dataclass
class IMDBData:
    train:         Split
    val:           Split
    test:          Split
    tokenizer:     SentimentTokenizer
    class_weights: dict[int, float]


class IMDBDataLoader:
    """Downloads IMDB, decodes to text, tokenises, and returns tf.data pipelines."""

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def load(self, batch_size: int = 64) -> IMDBData:
        log.info("Loading IMDB (num_words=%d) ...", self.cfg.vocab_size)
        X_train_raw, X_test_raw, y_train_all, y_test = self._download()

        tokenizer   = SentimentTokenizer(self.cfg.vocab_size, self.cfg.max_sequence_len)
        X_train_seq = tokenizer.fit_transform(X_train_raw)
        X_test_seq  = tokenizer.transform(X_test_raw)

        y_train_all = np.array(y_train_all, dtype=np.float32)
        y_test      = np.array(y_test,      dtype=np.float32)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_seq, y_train_all,
            test_size=self.cfg.val_split,
            stratify=y_train_all,
            random_state=self.cfg.random_seed,
        )

        class_weights = self._compute_class_weights(y_tr)

        return IMDBData(
            train=self._make_split(X_tr,  y_tr,  batch_size, shuffle=True),
            val=self._make_split(X_val, y_val, batch_size),
            test=self._make_split(X_test_seq, y_test, batch_size),
            tokenizer=tokenizer,
            class_weights=class_weights,
        )

    def _download(self) -> tuple[list[str], list[str], list[int], list[int]]:
        try:
            (X_tr_int, y_tr), (X_te_int, y_te) = imdb.load_data(num_words=self.cfg.vocab_size)
        except Exception as exc:
            raise DataError("Failed to download IMDB dataset") from exc

        idx2word = {v + 3: k for k, v in imdb.get_word_index().items()}
        idx2word |= {0: "<PAD>", 1: "<START>", 2: "<UNK>", 3: "<UNUSED>"}

        def decode(seq: np.ndarray) -> str:
            return " ".join(idx2word.get(i, "<UNK>") for i in seq)

        return (
            [decode(s) for s in X_tr_int],
            [decode(s) for s in X_te_int],
            list(y_tr), list(y_te),
        )

    @staticmethod
    def _make_split(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> Split:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False).cache().prefetch(AUTOTUNE)
        return Split(dataset=ds, size=len(X), pos_ratio=float(y.mean()))

    @staticmethod
    def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
        w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        return {0: float(w[0]), 1: float(w[1])}
```

---

## src/sentiment/models/base.py

```python
"""
base.py — Abstract base class for all sentiment models.
"""

from __future__ import annotations

import abc
from pathlib import Path

import tensorflow as tf
from sentiment.config import ModelConfig


class BaseSentimentModel(abc.ABC):

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg    = cfg
        self._model: tf.keras.Model | None = None

    @abc.abstractmethod
    def build(self, vocab_size: int, max_len: int) -> tf.keras.Model: ...

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise RuntimeError("Call .build() before accessing .model")
        return self._model

    def summary(self) -> None:
        self.model.summary(line_length=90)

    def save(self, path: str | Path, fmt: str = "tf") -> None:
        self.model.save(str(path), save_format=fmt)

    def predict_proba(self, X: object) -> object:
        return self.model.predict(X, verbose=0)

    def count_params(self) -> int:
        return int(self.model.count_params())
```

---

## src/sentiment/models/gru.py

```python
"""
gru.py — Bidirectional GRU with additive (Bahdanau) self-attention.

Architecture:
    Embedding -> SpatialDropout1D
    -> N x Bidirectional GRU (stacked, return_sequences=True)
    -> BahdanauAttention (optional)
    -> GlobalAvgPool + GlobalMaxPool -> Concat
    -> M x Dense (ReLU, L2, BatchNorm, Dropout)
    -> Dense(1, sigmoid)
"""

from __future__ import annotations

import logging
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, regularizers

from sentiment.config import ModelConfig, TrainingConfig
from sentiment.models.base import BaseSentimentModel

log = logging.getLogger(__name__)


class BahdanauAttention(layers.Layer):
    """
    Soft self-attention over the sequence axis.
    Returns a context vector weighted by learned alignment scores.
    """

    def __init__(self, units: int = 64, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, H: tf.Tensor, training: bool = False) -> tf.Tensor:
        # H: (batch, seq_len, hidden)
        score   = self.V(tf.nn.tanh(self.W(H)))       # (batch, seq_len, 1)
        alpha   = tf.nn.softmax(score, axis=1)         # (batch, seq_len, 1)
        context = tf.reduce_sum(alpha * H, axis=1)     # (batch, hidden)
        return context

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.W.units})
        return config


class GRUSentimentModel(BaseSentimentModel):
    """Production-grade Bidirectional GRU sentiment classifier."""

    def build(
        self,
        vocab_size:   int,
        max_len:      int,
        training_cfg: TrainingConfig | None = None,
    ) -> Model:
        cfg   = self.cfg
        t_cfg = training_cfg or TrainingConfig()
        reg   = regularizers.l2(cfg.l2_reg)

        inp = Input(shape=(max_len,), name="token_ids")

        x = layers.Embedding(vocab_size, cfg.embedding_dim, mask_zero=True, name="embedding")(inp)
        x = layers.SpatialDropout1D(cfg.spatial_dropout, name="spatial_drop")(x)

        for i, units in enumerate(cfg.gru_units):
            x = layers.Bidirectional(
                layers.GRU(
                    units,
                    return_sequences=True,
                    dropout=cfg.dropout_rate * 0.5,
                    recurrent_dropout=0.0,   # keep cuDNN-compatible
                    name=f"gru_{i}",
                ),
                name=f"bi_gru_{i}",
            )(x)

        if cfg.attention:
            attn = BahdanauAttention(units=cfg.gru_units[-1] * 2, name="attention")(x)
            avg  = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            max_ = layers.GlobalMaxPooling1D(name="max_pool")(x)
            x    = layers.Concatenate(name="pool_concat")([attn, avg, max_])
        else:
            avg  = layers.GlobalAveragePooling1D(name="avg_pool")(x)
            max_ = layers.GlobalMaxPooling1D(name="max_pool")(x)
            x    = layers.Concatenate(name="pool_concat")([avg, max_])

        for j, units in enumerate(cfg.dense_units):
            x = layers.Dense(units, activation="relu", kernel_regularizer=reg, name=f"dense_{j}")(x)
            x = layers.BatchNormalization(name=f"bn_{j}")(x)
            x = layers.Dropout(cfg.dropout_rate, name=f"drop_{j}")(x)

        out   = layers.Dense(1, activation="sigmoid", name="sentiment")(x)
        model = Model(inputs=inp, outputs=out, name="GRUSentiment")

        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=t_cfg.learning_rate,
                weight_decay=t_cfg.weight_decay,
            ),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=t_cfg.label_smoothing),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc",    curve="ROC"),
                tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

        self._model = model
        log.info("Model built — parameters: %s", f"{model.count_params():,}")
        return model
```

---

## src/sentiment/models/factory.py

```python
"""
factory.py — Model factory with registry pattern.
"""

from __future__ import annotations

from typing import Callable, ClassVar

from sentiment.config import ModelConfig
from sentiment.exceptions import ModelError
from sentiment.models.base import BaseSentimentModel
from sentiment.models.gru import GRUSentimentModel

ModelConstructor = Callable[[ModelConfig], BaseSentimentModel]


class ModelRegistry:
    _registry: ClassVar[dict[str, ModelConstructor]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[ModelConstructor], ModelConstructor]:
        def decorator(fn: ModelConstructor) -> ModelConstructor:
            cls._registry[name] = fn
            return fn
        return decorator

    @classmethod
    def create(cls, cfg: ModelConfig) -> BaseSentimentModel:
        if cfg.name not in cls._registry:
            raise ModelError(
                f"Unknown model '{cfg.name}'",
                context={"available": list(cls._registry)},
            )
        return cls._registry[cfg.name](cfg)

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._registry)


@ModelRegistry.register("gru_sentiment_v1")
def _build_gru_v1(cfg: ModelConfig) -> GRUSentimentModel:
    return GRUSentimentModel(cfg)
```

---

## src/sentiment/training/callbacks.py

```python
"""
callbacks.py — Custom Keras callbacks for production training.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import mlflow
import tensorflow as tf

log = logging.getLogger(__name__)


class MLflowCallback(tf.keras.callbacks.Callback):
    """Log epoch metrics + model artifact to MLflow."""

    def __init__(self, log_model: bool = True) -> None:
        super().__init__()
        self.log_model = log_model

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs:
            mlflow.log_metrics(logs, step=epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        if self.log_model and self.model is not None:
            mlflow.keras.log_model(self.model, artifact_path="model")
            log.info("Model logged to MLflow.")


class WarmupScheduler(tf.keras.callbacks.Callback):
    """Linear LR warm-up for the first `warmup_steps` batches."""

    def __init__(self, warmup_steps: int, target_lr: float) -> None:
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr    = target_lr
        self._step        = 0

    def on_train_batch_begin(self, batch: int, logs: Any = None) -> None:
        if self._step < self.warmup_steps:
            lr = self.target_lr * (self._step + 1) / self.warmup_steps
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self._step += 1


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Log wall-clock time per epoch."""

    def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
        elapsed = time.perf_counter() - self._t0
        log.info("Epoch %d — wall time: %.1fs", epoch + 1, elapsed)
        try:
            mlflow.log_metric("epoch_time_sec", elapsed, step=epoch)
        except Exception:
            pass
```

---

## src/sentiment/training/trainer.py

```python
"""
trainer.py — Orchestrates the full training lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import tensorflow as tf

from sentiment.config import AppConfig
from sentiment.data.dataset import IMDBData, IMDBDataLoader
from sentiment.models.factory import ModelRegistry
from sentiment.training.callbacks import EpochTimingCallback, MLflowCallback, WarmupScheduler

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    run_id:            str
    model_path:        Path
    tokenizer_path:    Path
    best_val_accuracy: float
    best_val_auc:      float
    history:           dict


class Trainer:
    """
    High-level training orchestrator.

    Usage::

        cfg    = AppConfig.from_yaml("configs/default.yaml")
        result = Trainer(cfg).run()
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._setup_logging()
        self._setup_mixed_precision()

    def run(self) -> TrainResult:
        mlflow.set_tracking_uri(self.cfg.logging.mlflow_uri)
        mlflow.set_experiment(self.cfg.logging.experiment)

        with mlflow.start_run() as run:
            mlflow.log_params(self._flat_params())
            data    = self._load_data()
            model   = self._build_model(data)
            history = self._train(model, data)
            paths   = self._save_artifacts(model, data)

        return TrainResult(
            run_id=run.info.run_id,
            model_path=paths["model"],
            tokenizer_path=paths["tokenizer"],
            best_val_accuracy=max(history.history["val_accuracy"]),
            best_val_auc=max(history.history.get("val_auc", [0.0])),
            history=history.history,
        )

    def _load_data(self) -> IMDBData:
        log.info("Loading dataset ...")
        return IMDBDataLoader(self.cfg.data).load(batch_size=self.cfg.training.batch_size)

    def _build_model(self, data: IMDBData) -> tf.keras.Model:
        log.info("Building model '%s' ...", self.cfg.model.name)
        sentiment_model = ModelRegistry.create(self.cfg.model)
        model = sentiment_model.build(
            vocab_size=data.tokenizer.effective_vocab_size,
            max_len=self.cfg.data.max_sequence_len,
            training_cfg=self.cfg.training,
        )
        model.summary(print_fn=log.info)
        return model

    def _train(self, model: tf.keras.Model, data: IMDBData) -> tf.keras.callbacks.History:
        return model.fit(
            data.train.dataset,
            validation_data=data.val.dataset,
            epochs=self.cfg.training.epochs,
            class_weight=data.class_weights,
            callbacks=self._build_callbacks(),
            verbose=1,
        )

    def _build_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        cfg = self.cfg
        t   = cfg.training
        p   = cfg.paths
        return [
            WarmupScheduler(t.lr_warmup_steps, t.learning_rate),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=t.early_stopping_patience,
                restore_best_weights=True, mode="max", verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(p.model_dir / "best_model"),
                monitor="val_auc", save_best_only=True, mode="max", verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=t.reduce_lr_factor,
                patience=t.reduce_lr_patience, min_lr=t.min_lr, verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=str(p.logs_dir), histogram_freq=1),
            MLflowCallback(log_model=cfg.logging.log_model),
            EpochTimingCallback(),
        ]

    def _save_artifacts(self, model: tf.keras.Model, data: IMDBData) -> dict[str, Path]:
        model_path     = self.cfg.paths.model_dir / "best_model"
        tokenizer_path = self.cfg.paths.model_dir / "tokenizer.pkl"
        data.tokenizer.save(tokenizer_path)
        log.info("Tokenizer saved -> %s", tokenizer_path)
        return {"model": model_path, "tokenizer": tokenizer_path}

    def _flat_params(self) -> dict[str, object]:
        cfg = self.cfg
        return {
            "vocab_size":      cfg.data.vocab_size,
            "max_len":         cfg.data.max_sequence_len,
            "model_name":      cfg.model.name,
            "embedding_dim":   cfg.model.embedding_dim,
            "gru_units":       str(cfg.model.gru_units),
            "attention":       cfg.model.attention,
            "batch_size":      cfg.training.batch_size,
            "epochs":          cfg.training.epochs,
            "learning_rate":   cfg.training.learning_rate,
            "label_smoothing": cfg.training.label_smoothing,
            "dropout_rate":    cfg.model.dropout_rate,
        }

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.cfg.logging.level.upper(), logging.INFO),
            format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _setup_mixed_precision(self) -> None:
        if self.cfg.training.mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            log.info("Mixed precision enabled (float16).")
```

---

## src/sentiment/serving/schemas.py

```python
"""
schemas.py — Pydantic v2 request/response models for the REST API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator

MAX_REVIEW_LEN = 5_000
MIN_REVIEW_LEN = 5


class ReviewRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=MIN_REVIEW_LEN,
        max_length=MAX_REVIEW_LEN,
        description="Raw movie review text.",
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Review text must not be blank.")
        return v.strip()


class BatchReviewRequest(BaseModel):
    reviews: list[ReviewRequest] = Field(..., min_length=1, max_length=64)


class SentimentResult(BaseModel):
    text:        str
    label:       str
    probability: float = Field(ge=0.0, le=1.0)
    confidence:  float = Field(ge=0.5, le=1.0)
    is_positive: bool
    latency_ms:  float | None = None


class BatchSentimentResult(BaseModel):
    results:        list[SentimentResult]
    total:          int
    n_positive:     int
    n_negative:     int
    avg_confidence: float
    latency_ms:     float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    version:      str


class ModelInfoResponse(BaseModel):
    model_name: str
    vocab_size: int
    max_len:    int
    n_params:   int
    framework:  str = "TensorFlow/Keras"
```

---

## src/sentiment/serving/predictor.py

```python
"""
predictor.py — Thread-safe, cached inference engine.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sentiment.config import APISettings
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.exceptions import ModelLoadError, ModelNotFoundError, ServingError
from sentiment.serving.schemas import BatchSentimentResult, SentimentResult

log = logging.getLogger(__name__)


class Predictor:
    """
    Stateful inference engine. Thread-safe; loads model once on construction.

    Example::

        predictor = Predictor.from_settings(APISettings())
        result    = predictor.predict("This movie was fantastic!")
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model:     tf.keras.Model,
        tokenizer: SentimentTokenizer,
        threshold: float = 0.5,
    ) -> None:
        self._model     = model
        self._tokenizer = tokenizer
        self.threshold  = threshold
        log.info("Predictor ready — threshold=%.2f", threshold)

    @classmethod
    def from_paths(
        cls,
        model_path:     str | Path,
        tokenizer_path: str | Path,
        threshold:      float = 0.5,
    ) -> "Predictor":
        model_path     = Path(model_path)
        tokenizer_path = Path(tokenizer_path)

        if not model_path.exists():
            raise ModelNotFoundError(
                "Saved model not found — run train.py first",
                context={"path": str(model_path)},
            )
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as exc:
            raise ModelLoadError("Failed to deserialise model") from exc

        tokenizer = SentimentTokenizer.load(tokenizer_path)
        return cls(model, tokenizer, threshold)

    @classmethod
    def from_settings(cls, settings: APISettings) -> "Predictor":
        return cls.from_paths(settings.model_path, settings.tokenizer_path)

    def predict(self, text: str) -> SentimentResult:
        t0   = time.perf_counter()
        prob = self._predict_batch([text])[0]
        ms   = (time.perf_counter() - t0) * 1000
        return self._build_result(text, prob, ms)

    def predict_batch(self, texts: list[str]) -> BatchSentimentResult:
        t0    = time.perf_counter()
        probs = self._predict_batch(texts)
        total_ms = (time.perf_counter() - t0) * 1000

        results = [self._build_result(t, p) for t, p in zip(texts, probs)]
        return BatchSentimentResult(
            results=results,
            total=len(results),
            n_positive=sum(1 for r in results if r.is_positive),
            n_negative=sum(1 for r in results if not r.is_positive),
            avg_confidence=float(np.mean([r.confidence for r in results])),
            latency_ms=round(total_ms, 2),
        )

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        try:
            with self._lock:
                X = self._tokenizer.transform(texts)
                return self._model.predict(X, verbose=0).flatten()
        except Exception as exc:
            raise ServingError("Inference failed") from exc

    def _build_result(self, text: str, prob: float, latency_ms: float | None = None) -> SentimentResult:
        is_pos = bool(prob >= self.threshold)
        conf   = float(prob if is_pos else 1 - prob)
        return SentimentResult(
            text=text,
            label="positive" if is_pos else "negative",
            probability=float(prob),
            confidence=conf,
            is_positive=is_pos,
            latency_ms=round(latency_ms, 2) if latency_ms else None,
        )
```

---

## src/sentiment/serving/api.py

```python
"""
api.py — Production FastAPI REST service.

Endpoints:
    GET  /health          - liveness probe
    GET  /model/info      - model metadata
    POST /predict         - single review
    POST /predict/batch   - batch reviews (max 64)

Extras: Prometheus metrics, structured logging, async-safe design.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from sentiment.config import APISettings
from sentiment.exceptions import InvalidInputError, ModelNotFoundError, ServingError
from sentiment.serving.predictor import Predictor
from sentiment.serving.schemas import (
    BatchReviewRequest, BatchSentimentResult,
    HealthResponse, ModelInfoResponse,
    ReviewRequest, SentimentResult,
)

log = logging.getLogger(__name__)

REQUEST_COUNT   = Counter("sentiment_requests_total", "Total prediction requests", ["endpoint", "label"])
REQUEST_LATENCY = Histogram("sentiment_request_latency_seconds", "Prediction latency", ["endpoint"])


def create_app(settings: APISettings | None = None) -> FastAPI:
    settings = settings or APISettings()
    predictor_store: dict[str, Predictor] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        log.info("Loading model from %s ...", settings.model_path)
        try:
            predictor_store["main"] = Predictor.from_settings(settings)
            log.info("Predictor ready.")
        except ModelNotFoundError:
            log.critical("Model not found — start server after training!")
        yield
        predictor_store.clear()

    app = FastAPI(
        title="GRU Sentiment Analysis API",
        description="Bidirectional GRU sentiment classifier for IMDB reviews.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def get_predictor() -> Predictor:
        p = predictor_store.get("main")
        if p is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
        return p

    @app.middleware("http")
    async def log_latency(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0       = time.perf_counter()
        response = await call_next(request)
        ms       = (time.perf_counter() - t0) * 1000
        log.info("%s %s - %dms", request.method, request.url.path, round(ms))
        return response

    @app.exception_handler(ServingError)
    async def serving_error_handler(request: Request, exc: ServingError) -> JSONResponse:
        return JSONResponse(status_code=500, content={"error": str(exc), "context": exc.context})

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded="main" in predictor_store, version="1.0.0")

    @app.get("/metrics", tags=["System"])
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
    async def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=predictor._tokenizer.__class__.__name__,
            vocab_size=predictor._tokenizer.vocab_size,
            max_len=predictor._tokenizer.max_len,
            n_params=int(predictor._model.count_params()),
        )

    @app.post("/predict", response_model=SentimentResult, tags=["Prediction"])
    async def predict(body: ReviewRequest, predictor: Predictor = Depends(get_predictor)) -> SentimentResult:
        with REQUEST_LATENCY.labels(endpoint="/predict").time():
            result = predictor.predict(body.text)
        REQUEST_COUNT.labels(endpoint="/predict", label=result.label).inc()
        return result

    @app.post("/predict/batch", response_model=BatchSentimentResult, tags=["Prediction"])
    async def predict_batch(body: BatchReviewRequest, predictor: Predictor = Depends(get_predictor)) -> BatchSentimentResult:
        texts  = [r.text for r in body.reviews]
        with REQUEST_LATENCY.labels(endpoint="/predict/batch").time():
            result = predictor.predict_batch(texts)
        for r in result.results:
            REQUEST_COUNT.labels(endpoint="/predict/batch", label=r.label).inc()
        return result

    return app


def start() -> None:
    settings = APISettings()
    uvicorn.run("sentiment.serving.api:create_app", factory=True,
                host=settings.host, port=settings.port,
                workers=settings.workers, log_level=settings.log_level)

if __name__ == "__main__":
    start()
```

---

## scripts/train.py — Typer CLI

```python
"""
scripts/train.py — Training entry-point with Typer + Rich output.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 20 --lr 5e-4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentiment.config import AppConfig
from sentiment.training.trainer import Trainer

app     = typer.Typer(help="GRU Sentiment — Training CLI", add_completion=False)
console = Console()


@app.command()
def train(
    config: Path         = typer.Option(Path("configs/default.yaml"), help="YAML config path"),
    epochs: Optional[int]   = typer.Option(None, help="Override training epochs"),
    lr:     Optional[float] = typer.Option(None, help="Override learning rate"),
    batch:  Optional[int]   = typer.Option(None, help="Override batch size"),
) -> None:
    """Train the GRU sentiment model from scratch."""
    console.rule("[bold cyan]GRU Sentiment — Training")
    cfg = AppConfig.from_yaml(config)

    if epochs: cfg.training.epochs          = epochs
    if lr:     cfg.training.learning_rate   = lr
    if batch:  cfg.training.batch_size      = batch

    table = Table(title="Active Configuration", header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value",     style="green")
    for k, v in [
        ("Model", cfg.model.name), ("Epochs", cfg.training.epochs),
        ("Batch Size", cfg.training.batch_size), ("LR", cfg.training.learning_rate),
        ("GRU Units", cfg.model.gru_units), ("Attention", cfg.model.attention),
        ("MLflow URI", cfg.logging.mlflow_uri),
    ]:
        table.add_row(str(k), str(v))
    console.print(table)

    with console.status("[bold green]Training in progress ..."):
        result = Trainer(cfg).run()

    console.print(Panel(
        f"[bold green]Training complete![/]\n\n"
        f"  Run ID       : [cyan]{result.run_id}[/]\n"
        f"  Val Accuracy : [yellow]{result.best_val_accuracy * 100:.2f}%[/]\n"
        f"  Val AUC      : [yellow]{result.best_val_auc:.4f}[/]\n"
        f"  Model        : [blue]{result.model_path}[/]",
        title="Results", border_style="green",
    ))


if __name__ == "__main__":
    app()
```

---

## tests/conftest.py

```python
"""conftest.py — Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from sentiment.config import AppConfig, ModelConfig, TrainingConfig
from sentiment.data.preprocessing import SentimentTokenizer
from sentiment.models.gru import GRUSentimentModel
from sentiment.serving.predictor import Predictor

SAMPLE_REVIEWS = [
    "This film was absolutely brilliant and moving.",
    "Terrible waste of time. Walked out after 20 minutes.",
    "A perfectly average thriller with highs and lows.",
] * 20


@pytest.fixture(scope="session")
def default_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.data.vocab_size       = 500
    cfg.data.max_sequence_len = 50
    cfg.model.gru_units       = [16]
    cfg.model.dense_units     = [8]
    cfg.training.epochs       = 1
    cfg.training.batch_size   = 4
    return cfg


@pytest.fixture(scope="session")
def fitted_tokenizer(default_cfg: AppConfig) -> SentimentTokenizer:
    tok = SentimentTokenizer(vocab_size=default_cfg.data.vocab_size, max_len=default_cfg.data.max_sequence_len)
    tok.fit(SAMPLE_REVIEWS)
    return tok


@pytest.fixture(scope="session")
def built_model(default_cfg: AppConfig) -> tf.keras.Model:
    return GRUSentimentModel(default_cfg.model).build(
        vocab_size=default_cfg.data.vocab_size,
        max_len=default_cfg.data.max_sequence_len,
        training_cfg=default_cfg.training,
    )


@pytest.fixture(scope="session")
def predictor(built_model: tf.keras.Model, fitted_tokenizer: SentimentTokenizer) -> Predictor:
    return Predictor(model=built_model, tokenizer=fitted_tokenizer)
```

---

## tests/unit/test_preprocessing.py

```python
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
```

---

## tests/unit/test_model.py

```python
"""Unit tests for GRU model construction and forward pass."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from sentiment.config import ModelConfig, TrainingConfig
from sentiment.exceptions import ModelError
from sentiment.models.factory import ModelRegistry
from sentiment.models.gru import BahdanauAttention, GRUSentimentModel


class TestBahdanauAttention:
    def test_output_shape(self) -> None:
        H   = tf.random.normal((8, 20, 64))
        out = BahdanauAttention(units=32)(H)
        assert out.shape == (8, 64)


class TestGRUSentimentModel:
    @pytest.fixture
    def small_model(self) -> tf.keras.Model:
        cfg             = ModelConfig()
        cfg.gru_units   = [16]
        cfg.dense_units = [8]
        cfg.attention   = True
        return GRUSentimentModel(cfg).build(vocab_size=500, max_len=50)

    def test_output_shape(self, small_model: tf.keras.Model) -> None:
        X = np.random.randint(0, 500, (4, 50))
        assert small_model.predict(X, verbose=0).shape == (4, 1)

    def test_output_in_01(self, small_model: tf.keras.Model) -> None:
        X   = np.random.randint(0, 500, (4, 50))
        out = small_model.predict(X, verbose=0).flatten()
        assert np.all((out >= 0) & (out <= 1))

    def test_registry_resolves(self) -> None:
        assert isinstance(ModelRegistry.create(ModelConfig()), GRUSentimentModel)

    def test_unknown_model_raises(self) -> None:
        cfg      = ModelConfig()
        cfg.name = "does_not_exist"
        with pytest.raises(ModelError):
            ModelRegistry.create(cfg)
```

---

## tests/integration/test_api.py

```python
"""Integration tests for the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentiment.serving.api import create_app
from sentiment.serving.predictor import Predictor


@pytest.fixture
def client(predictor: Predictor) -> TestClient:
    app = create_app()

    from sentiment.serving import api as api_module

    app.dependency_overrides[api_module.get_predictor] = lambda: predictor  # type: ignore[attr-defined]
    return TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200


class TestPredictEndpoint:
    def test_valid_review(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": "This movie was absolutely fantastic!"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"positive", "negative"}
        assert 0.0 <= body["probability"] <= 1.0
        assert body["confidence"] >= 0.5

    def test_blank_text_rejected(self, client: TestClient) -> None:
        assert client.post("/predict", json={"text": "   "}).status_code == 422

    def test_too_short_rejected(self, client: TestClient) -> None:
        assert client.post("/predict", json={"text": "ok"}).status_code == 422

    def test_batch_predict(self, client: TestClient) -> None:
        payload = {"reviews": [
            {"text": "Incredible masterpiece, deeply moving."},
            {"text": "Boring and predictable. Complete disappointment."},
        ]}
        r    = client.post("/predict/batch", json=payload)
        body = r.json()
        assert r.status_code == 200
        assert body["total"] == 2
        assert body["n_positive"] + body["n_negative"] == 2

    def test_batch_over_limit_rejected(self, client: TestClient) -> None:
        payload = {"reviews": [{"text": "Some review text here."}] * 65}
        assert client.post("/predict/batch", json=payload).status_code == 422
```

---

## docker/docker-compose.yml

```yaml
version: "3.9"

services:

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.api
    image: sentiment-api:latest
    ports:
      - "8000:8000"
    environment:
      SENTIMENT_MODEL_PATH:      /artifacts/models/best_model
      SENTIMENT_TOKENIZER_PATH:  /artifacts/models/tokenizer.pkl
      SENTIMENT_WORKERS:         "2"
      SENTIMENT_LOG_LEVEL:       "info"
    volumes:
      - ../artifacts:/artifacts:ro
    healthcheck:
      test:     ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout:  5s
      retries:  3
    restart: unless-stopped

  app:
    build:
      context: ..
      dockerfile: docker/Dockerfile.app
    image: sentiment-app:latest
    ports:
      - "8501:8501"
    environment:
      SENTIMENT_API_URL: http://api:8000
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.1
    ports:
      - "5000:5000"
    command: >
      mlflow server
        --host 0.0.0.0
        --port 5000
        --backend-store-uri sqlite:///mlruns.db
        --default-artifact-root /mlartifacts
    volumes:
      - mlflow_data:/mlartifacts
      - mlflow_db:/mlruns.db
    restart: unless-stopped

volumes:
  mlflow_data:
  mlflow_db:
```

---

## docker/Dockerfile.api

```dockerfile
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --upgrade pip && pip install ".[dev]" --target /install

FROM base AS runtime
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY src/     /app/src/
COPY scripts/ /app/scripts/
COPY configs/ /app/configs/
ENV PYTHONPATH=/app/src

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "sentiment.serving.api"]
```

---

## .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install ruff mypy pydantic-settings
      - run: ruff check src/ scripts/ tests/
      - run: mypy src/sentiment --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    needs: lint-and-type-check
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --cov=src/sentiment --cov-report=xml -q
      - uses: codecov/codecov-action@v4
        with: { files: coverage.xml }
```

---

## .github/workflows/cd.yml

```yaml
name: CD - Build and Push Docker

on:
  push:
    branches: [main]

env:
  REGISTRY:   ghcr.io
  IMAGE_BASE: ${{ github.repository_owner }}/sentiment

jobs:
  build-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry:  ${{ env.REGISTRY }}
          username:  ${{ github.actor }}
          password:  ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context:    .
          file:       docker/Dockerfile.api
          push:       true
          tags:       ${{ env.REGISTRY }}/${{ env.IMAGE_BASE }}-api:latest
      - uses: docker/build-push-action@v5
        with:
          context:    .
          file:       docker/Dockerfile.app
          push:       true
          tags:       ${{ env.REGISTRY }}/${{ env.IMAGE_BASE }}-app:latest
```

---

## Makefile

```makefile
.PHONY: install train evaluate serve app test lint type-check docker-up docker-down clean

install:
	pip install -e ".[dev]"
	pre-commit install

train:
	python scripts/train.py --config configs/default.yaml

train-large:
	python scripts/train.py --config configs/experiment_gru_large.yaml

evaluate:
	python scripts/evaluate.py

serve:
	uvicorn "sentiment.serving.api:create_app" --factory \
	  --host 0.0.0.0 --port 8000 --reload

app:
	streamlit run app/streamlit_app.py

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

test:
	pytest tests/ -v --cov=src/sentiment --cov-report=term-missing

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

type-check:
	mypy src/sentiment --ignore-missing-imports

docker-up:
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
```

---

## Senior-Level Upgrade Summary

| Concern | Junior Version | Senior Version |
|---|---|---|
| **Config** | Hard-coded constants | Pydantic v2 typed config + YAML + env-var override |
| **Exceptions** | Generic `Exception` | Typed hierarchy with `context` dict |
| **Logging** | `print()` everywhere | Structured `logging` module + timestamps |
| **Data pipeline** | `model.fit(X, y)` array | `tf.data` with caching, prefetching, stratified split |
| **Model** | Single GRU | Stacked Bi-GRU + Bahdanau attention + L2 + label smoothing |
| **Training** | Inline script | `Trainer` orchestrator class + `ModelRegistry` factory |
| **Experiment tracking** | None | MLflow (params, metrics, model artifact, run ID) |
| **LR schedule** | None | Linear warm-up + `ReduceLROnPlateau` + AdamW |
| **Serving** | Streamlit only | FastAPI REST + Prometheus metrics + Streamlit calling the API |
| **Input validation** | None | Pydantic request/response schemas with field validators |
| **Testing** | None | pytest unit + integration tests with ≥80% coverage gate |
| **Type safety** | None | `mypy --strict` annotations throughout |
| **Linting** | None | `ruff` + `pre-commit` hooks |
| **Deployment** | Manual | Docker Compose (API + App + MLflow) with health-checks |
| **CI/CD** | None | GitHub Actions: lint → test → build → push GHCR |
| **CLI** | `argparse` | `Typer` + `Rich` formatted output |
| **Package structure** | Flat scripts | `src/` layout with proper sub-packages |
| **Class weights** | None | `compute_class_weight` for imbalanced data |
| **Reproducibility** | None | `random_seed` threaded through all splits |
