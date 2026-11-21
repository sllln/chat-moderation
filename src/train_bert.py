"""
Step 7 - Fine-tune BERTurk for offensive language classification (technique area C).

BERTurk (dbmdz/bert-base-turkish-cased) is a BERT model that was already
pre-trained on a large Turkish corpus. We add a small classification head
on top of it and train the whole model for a few epochs on our tweets.

  - Text is cleaned with clean_light (keeps case and Turkish letters).
  - The WordPiece tokenizer of BERTurk splits words into sub-word pieces.
  - The best epoch is chosen by macro-F1 on the VALIDATION set.
  - Uses the GPU if one is available (mixed precision, fp16).

Saves the fine-tuned model to models/berturk/
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from text_cleaning import clean_light

BASE_MODEL = "dbmdz/bert-base-turkish-cased"
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUT_DIR = MODELS_DIR / "berturk"
CHECKPOINT_DIR = MODELS_DIR / "berturk_checkpoints"

MAX_LENGTH = 128      # longer messages are cut off after 128 tokens
BATCH_SIZE = 16       # lower to 8 if the GPU runs out of memory
EPOCHS = 3
LEARNING_RATE = 2e-5  # small steps: we only adjust a model that already knows Turkish

print(f"GPU available: {torch.cuda.is_available()}"
      + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def load_split(name: str) -> Dataset:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    frame["text"] = frame["text"].map(clean_light)
    dataset = Dataset.from_pandas(frame[["text", "label"]])
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH),
        batched=True,
    )


train = load_split("train")
validation = load_split("validation")

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=2,
    id2label={0: "clean", 1: "offensive"},
    label2id={"clean": 0, "offensive": 1},
)


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, predictions, average="macro")}


args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    warmup_steps=0.1,                  # first 10% of the steps: slowly increase the learning rate
    weight_decay=0.01,
    eval_strategy="epoch",             # measure validation macro-F1 after every epoch
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=torch.cuda.is_available(),    # half precision on the GPU: less memory, faster
    logging_steps=100,
    report_to="none",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=validation,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),  # pads each batch only to its longest message
    compute_metrics=compute_metrics,
)

trainer.train()
best = trainer.evaluate()
print(f"\nBest validation macro-F1: {best['eval_macro_f1']:.3f}")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)   # checkpoints are large and no longer needed

size_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*")) / 1024 / 1024
print(f"Saved to {OUTPUT_DIR} ({size_mb:.0f} MB)")