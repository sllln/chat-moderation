"""
Step 8 - Quantize the fine-tuned BERTurk model (technique area D).

  - Loads the 32-bit model from models/berturk/
  - Applies dynamic INT8 quantization (see bert_model.py)
  - Saves it to models/berturk_int8.pt
  - Compares size, speed and validation macro-F1 of both versions on the CPU
"""
import time

import pandas as pd
import torch
from sklearn.metrics import f1_score

from bert_model import FP32_DIR, INT8_FILE, load_fp32, load_int8, load_tokenizer, predict_proba, quantize

torch.manual_seed(42)

print("Quantizing ...")
quantized = quantize(load_fp32())
torch.save(quantized.state_dict(), INT8_FILE)

fp32_mb = (FP32_DIR / "model.safetensors").stat().st_size / 1024 / 1024
int8_mb = INT8_FILE.stat().st_size / 1024 / 1024
print(f"FP32 model size: {fp32_mb:.0f} MB")
print(f"INT8 model size: {int8_mb:.0f} MB  ({fp32_mb / int8_mb:.1f}x smaller)")

validation = pd.read_csv("data/validation.csv")
texts, labels = validation["text"].tolist(), validation["label"].tolist()
tokenizer = load_tokenizer()

print(f"\nValidation set ({len(texts)} tweets) on the CPU:")
for name, model in [("FP32", load_fp32()), ("INT8", load_int8())]:
    start = time.perf_counter()
    probabilities = predict_proba(model, tokenizer, texts)
    seconds = time.perf_counter() - start
    predictions = [int(p >= 0.5) for p in probabilities]
    f1 = f1_score(labels, predictions, average="macro")
    print(f"  {name}: macro-F1 = {f1:.3f}   time = {seconds:.1f}s")