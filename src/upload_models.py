"""
Uploads the trained models in models/ to the Hugging Face Hub (run once, by the author).

Before running:  hf auth login   (paste a token with "Write" permission)
"""
from pathlib import Path

from huggingface_hub import HfApi

from download_models import HF_REPO_ID, MODELS_DIR, missing_files

MODEL_CARD = """---
language: tr
license: mit
tags:
  - text-classification
  - offensive-language
  - turkish
  - quantization
datasets:
  - coltekin/offenseval2020_tr
base_model: dbmdz/bert-base-turkish-cased
---

# Turkish chat moderation models (SWE015 homework)

Models for the real-time chat moderation app at https://github.com/sllln/chat-moderation.
All of them classify a Turkish message as **clean (0)** or **offensive (1)**.

| File | Model |
|---|---|
| `berturk/` | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned on OffensEval-TR 2020, FP32 |
| `berturk_int8.pt` | The same model after dynamic INT8 quantization (PyTorch, Linear layers) |
| `tfidf_logreg.joblib` | tf-idf (word 1-2 grams + char 2-5 grams) + logistic regression |
| `word2vec.model*` | word2vec (skip-gram, 100 dim) trained on the training tweets |
| `ffnn.pt` | Feedforward network on averaged + max-pooled word2vec vectors |

Training data: OffensEval-TR 2020 (Çöltekin, 2020). Results on the official test set
are in the GitHub repository. Intended for education and research only; the models
make mistakes and can reflect biases of the training data.

`berturk_int8.pt` and `tfidf_logreg.joblib` are pickle-based files: only load them
from this repository.
"""

if missing_files():
    raise SystemExit(f"Train the models first; missing: {missing_files()}")

api = HfApi()
api.create_repo(HF_REPO_ID, repo_type="model", private=False, exist_ok=True)
print(f"Uploading {MODELS_DIR}/ to https://huggingface.co/{HF_REPO_ID} ...")
api.upload_folder(
    folder_path=MODELS_DIR,
    repo_id=HF_REPO_ID,
    repo_type="model",
    ignore_patterns=["berturk_checkpoints/*", "*/training_args.bin"],
    commit_message="Upload trained models",
)
api.upload_file(
    path_or_fileobj=MODEL_CARD.encode("utf-8"),
    path_in_repo="README.md",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Add model card",
)
print("Done.")
