"""
Step 1 - Explore the OffensEval-TR 2020 dataset.

Downloads the dataset from Hugging Face and prints its structure
(splits, columns, labels, example tweets) so that we know exactly
what we are working with before writing any model code.
"""
from collections import Counter

from datasets import ClassLabel, load_dataset

DATASET_ID = "coltekin/offenseval2020_tr"

print(f"Downloading {DATASET_ID} ...")
dataset = load_dataset(DATASET_ID, revision="refs/convert/parquet")

print("\n=== Dataset overview ===")
print(dataset)

print("\n=== Columns and their types ===")
print(dataset["train"].features)

print("\n=== Label distribution per split ===")
for split_name, split in dataset.items():
    for column, feature in split.features.items():
        if isinstance(feature, ClassLabel):
            counts = Counter(split[column])
            readable = {feature.int2str(k): v for k, v in sorted(counts.items())}
            print(f"{split_name} / {column}: {readable}")

print("\n=== First 5 training examples ===")
for row in dataset["train"].select(range(5)):
    print(row)