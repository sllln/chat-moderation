"""
Downloads the trained models from the Hugging Face Hub into models/.

The models are too large for GitHub (requirement R7), so they are stored at
https://huggingface.co/selliinn/chat-moderation-tr-models

ensure_models() is called automatically by the app and by the evaluation;
it only downloads when a file is missing. You can also run this file directly:
    python src/download_models.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "selliinn/chat-moderation-tr-models"
MODELS_DIR = Path("models")

REQUIRED_FILES = [
    "berturk/config.json",
    "berturk/model.safetensors",
    "berturk/tokenizer_config.json",
    "berturk_int8.pt",
    "tfidf_logreg.joblib",
    "word2vec.model",
    "ffnn.pt",
]


def missing_files() -> list[str]:
    return [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]


def ensure_models() -> None:
    missing = missing_files()
    if not missing:
        return
    print(f"Downloading models from https://huggingface.co/{HF_REPO_ID} (about 630 MB, only once) ...")
    snapshot_download(repo_id=HF_REPO_ID, local_dir=MODELS_DIR)
    still_missing = missing_files()
    if still_missing:
        raise FileNotFoundError(f"These model files are still missing after the download: {still_missing}")
    print("Models ready.")


if __name__ == "__main__":
    ensure_models()
