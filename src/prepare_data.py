"""
Step 2 - Prepare the data splits.

Takes the official OffensEval-TR 2020 train/test splits and:
  - carves a 10% validation set out of the training data (stratified),
  - keeps the official test set untouched,
  - saves all three splits as CSV files in the data/ folder.

Labels: 0 = NOT (clean), 1 = OFF (offensive)
"""
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET_ID = "coltekin/offenseval2020_tr"
REVISION = "refs/convert/parquet"
DATA_DIR = Path("data")
VALIDATION_SIZE = 0.10
RANDOM_SEED = 42

DATA_DIR.mkdir(exist_ok=True)

dataset = load_dataset(DATASET_ID, revision=REVISION)

column_names = {"tweet": "text", "subtask_a": "label"}
full_train = dataset["train"].to_pandas().rename(columns=column_names)
test = dataset["test"].to_pandas().rename(columns=column_names)

train, validation = train_test_split(
    full_train,
    test_size=VALIDATION_SIZE,
    stratify=full_train["label"],
    random_state=RANDOM_SEED,
)

print()
for name, frame in [("train", train), ("validation", validation), ("test", test)]:
    path = DATA_DIR / f"{name}.csv"
    frame.to_csv(path, index=False, encoding="utf-8")
    offensive_share = frame["label"].mean() * 100
    print(f"{name:<10} {len(frame):>6} rows | offensive: {offensive_share:.1f}% | saved to {path}")

overlap = set(test["text"]) & set(full_train["text"])
print(f"\nTest tweets that also appear in the training data: {len(overlap)}")