"""
One common interface for all five models. Used by the evaluation and by the app.

Every classifier has:
  name                      - shown in the results table and in the app
  precision_label           - FP32 / INT8 / "rules" ... (for the results table)
  predict_proba(text)       - probability that ONE message is offensive (0.0 - 1.0)
  predict_proba_many(texts) - the same for a list of messages
  size_mb()                 - size of the files the model needs on disk
"""
from pathlib import Path

import joblib
import numpy as np
import torch
from gensim.models import Word2Vec

import bert_model
from download_models import ensure_models
from lexicon_matcher import LexiconMatcher
from neural_model import FeedForwardNet, message_vector

DATA_DIR = Path("data")
MODELS_DIR = Path("models")


class BaseClassifier:
    name = ""
    precision_label = ""
    files: list[Path] = []

    def predict_proba(self, text: str) -> float:
        raise NotImplementedError

    def predict_proba_many(self, texts: list[str]) -> list[float]:
        return [self.predict_proba(t) for t in texts]

    def size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.files) / 1024 / 1024


class LexiconClassifier(BaseClassifier):
    name = "Lexicon + edit distance"
    precision_label = "rules"

    def __init__(self):
        self.files = [DATA_DIR / "lexicon_selected.txt", DATA_DIR / "safe_words.txt"]
        self.matcher = LexiconMatcher.from_files(*self.files)

    def predict_proba(self, text: str) -> float:
        # A rule either fires or not: 1.0 or 0.0
        return 1.0 if self.matcher.find_matches(text) else 0.0


class TfidfClassifier(BaseClassifier):
    name = "tf-idf + logistic regression"
    precision_label = "FP64"

    def __init__(self):
        self.files = [MODELS_DIR / "tfidf_logreg.joblib"]
        self.model = joblib.load(self.files[0])

    def predict_proba(self, text: str) -> float:
        return float(self.model.predict_proba([text])[0, 1])

    def predict_proba_many(self, texts: list[str]) -> list[float]:
        return self.model.predict_proba(texts)[:, 1].tolist()


class FeedForwardClassifier(BaseClassifier):
    name = "word2vec + feedforward NN"
    precision_label = "FP32"

    def __init__(self):
        # word2vec may store its vectors in extra .npy files next to the .model file
        self.files = sorted(MODELS_DIR.glob("word2vec.model*")) + [MODELS_DIR / "ffnn.pt"]
        self.word_vectors = Word2Vec.load(str(MODELS_DIR / "word2vec.model")).wv
        self.net = FeedForwardNet(input_size=2 * self.word_vectors.vector_size)
        self.net.load_state_dict(torch.load(MODELS_DIR / "ffnn.pt"))
        self.net.eval()

    @torch.inference_mode()
    def predict_proba(self, text: str) -> float:
        x = torch.from_numpy(message_vector(text, self.word_vectors)).unsqueeze(0)
        return float(torch.sigmoid(self.net(x))[0])

    @torch.inference_mode()
    def predict_proba_many(self, texts: list[str]) -> list[float]:
        x = torch.from_numpy(np.stack([message_vector(t, self.word_vectors) for t in texts]))
        return torch.sigmoid(self.net(x)).tolist()


class BertClassifier(BaseClassifier):
    def __init__(self, quantized: bool):
        self.tokenizer = bert_model.load_tokenizer()
        if quantized:
            self.name = "BERTurk (quantized)"
            self.precision_label = "INT8 (dynamic)"
            self.files = [bert_model.INT8_FILE]
            self.model = bert_model.load_int8()
        else:
            self.name = "BERTurk"
            self.precision_label = "FP32"
            self.files = [bert_model.FP32_DIR / "model.safetensors"]
            self.model = bert_model.load_fp32()

    def predict_proba(self, text: str) -> float:
        return bert_model.predict_proba(self.model, self.tokenizer, [text])[0]

    def predict_proba_many(self, texts: list[str]) -> list[float]:
        return bert_model.predict_proba(self.model, self.tokenizer, texts)


def load_all_classifiers() -> list[BaseClassifier]:
    """All five models, from the simplest to the most advanced."""
    ensure_models()   # downloads the models from the Hugging Face Hub if they are missing
    return [
        LexiconClassifier(),
        TfidfClassifier(),
        FeedForwardClassifier(),
        BertClassifier(quantized=False),
        BertClassifier(quantized=True),
    ]
