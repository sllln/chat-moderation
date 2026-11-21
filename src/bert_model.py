"""
Helpers for the fine-tuned BERTurk model (technique areas C and D).

  - load_tokenizer() / load_fp32() : the fine-tuned model in normal 32-bit precision
  - quantize(model)                : dynamic INT8 quantization
  - load_int8()                    : the saved quantized model
  - predict_proba(...)             : probability of "offensive" for a list of messages

Everything here runs on the CPU, as required by the use case.
"""
import warnings
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from text_cleaning import clean_light

MODELS_DIR = Path("models")
FP32_DIR = MODELS_DIR / "berturk"
INT8_FILE = MODELS_DIR / "berturk_int8.pt"
MAX_LENGTH = 128


def quantize(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of every Linear layer.

    The weights are stored as 8-bit integers instead of 32-bit floats; the
    inputs of each layer are converted to 8 bits on the fly while the model
    runs. This makes the model smaller and faster on the CPU.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # PyTorch prints deprecation notes for this API
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(FP32_DIR)


def load_fp32() -> nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(FP32_DIR).eval()


def load_int8() -> nn.Module:
    # Build the model structure from the config, turn it into the quantized
    # structure, then fill it with the saved INT8 weights.
    config = AutoConfig.from_pretrained(FP32_DIR)
    model = quantize(AutoModelForSequenceClassification.from_config(config).eval())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # weights_only=False: quantized weights are stored as special objects.
        # Only do this with files you created yourself.
        state = torch.load(INT8_FILE, weights_only=False)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 64) -> list[float]:
    """Probability of "offensive" (label 1) for every message."""
    probabilities = []
    for start in range(0, len(texts), batch_size):
        batch = [clean_light(t) for t in texts[start:start + batch_size]]
        encoded = tokenizer(batch, truncation=True, max_length=MAX_LENGTH,
                            padding=True, return_tensors="pt")
        logits = model(**encoded).logits
        probabilities += torch.softmax(logits, dim=-1)[:, 1].tolist()
    return probabilities