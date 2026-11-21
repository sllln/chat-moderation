"""
Step 9 - Final evaluation of all models on the held-out TEST set (requirements R3 + R4).

For every model:
  - task metric : macro-F1 on the test set (plus precision / recall of "offensive")
  - size        : MB on disk
  - latency     : time to classify ONE message on the CPU, averaged over
                  LATENCY_RUNS different test messages (after a short warm-up)

Writes:
  results/results_table.csv     - the results table
  results/results_table.md      - the same table, ready to copy into the slides
  results/test_predictions.csv  - every model's prediction for every test tweet
"""
import platform
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from classifiers import load_all_classifiers

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
LATENCY_RUNS = 200
WARMUP_RUNS = 10
LATENCY_BUDGET_MS = 50


def cpu_name() -> str:
    """Readable CPU name (Windows registry), with a generic fallback."""
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor() or platform.machine()


RESULTS_DIR.mkdir(exist_ok=True)
test = pd.read_csv(DATA_DIR / "test.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
texts, labels = test["text"].tolist(), test["label"].tolist()

hardware = f"{cpu_name()}, CPU only, {torch.get_num_threads()} PyTorch threads"
print(f"Hardware: {hardware}")
print(f"Test set: {len(texts)} tweets\n")

latency_messages = test["text"].sample(LATENCY_RUNS, random_state=42).tolist()
warmup_messages = validation["text"].sample(WARMUP_RUNS, random_state=42).tolist()

rows = []
predictions_table = test[["id", "text", "label"]].copy()
for model in load_all_classifiers():
    print(f"--- {model.name} ---")

    # 1) Latency: one message at a time, like in the chat app.
    for message in warmup_messages:
        model.predict_proba(message)
    timings = []
    for message in latency_messages:
        start = time.perf_counter()
        model.predict_proba(message)
        timings.append((time.perf_counter() - start) * 1000)
    mean_ms = statistics.mean(timings)
    p95_ms = sorted(timings)[int(0.95 * len(timings)) - 1]

    # 2) Quality on the whole test set.
    start = time.perf_counter()
    probabilities = model.predict_proba_many(texts)
    predictions = [int(p >= 0.5) for p in probabilities]
    predictions_table[model.name] = predictions

    row = {
        "model": model.name,
        "numeric_precision": model.precision_label,
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "offensive_precision": precision_score(labels, predictions),
        "offensive_recall": recall_score(labels, predictions),
        "size_mb": model.size_mb(),
        "latency_mean_ms": mean_ms,
        "latency_p95_ms": p95_ms,
        "within_budget": mean_ms <= LATENCY_BUDGET_MS,
    }
    rows.append(row)
    print(f"macro-F1 {row['macro_f1']:.3f} | {row['size_mb']:.1f} MB | "
          f"{mean_ms:.2f} ms/message (p95 {p95_ms:.2f}) | evaluated in {time.perf_counter() - start:.0f}s\n")

table = pd.DataFrame(rows)
table.to_csv(RESULTS_DIR / "results_table.csv", index=False)
predictions_table.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

markdown = [
    f"Test set: OffensEval-TR 2020 official test split ({len(texts)} tweets). "
    f"Latency: mean over {LATENCY_RUNS} single-message runs. Hardware: {hardware}.",
    "",
    "| Model | Numeric precision | Macro-F1 | Size (MB) | Latency mean (ms) | Latency p95 (ms) | <= 50 ms |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    markdown.append(
        f"| {r['model']} | {r['numeric_precision']} | {r['macro_f1']:.3f} | {r['size_mb']:.1f} | "
        f"{r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} | {'yes' if r['within_budget'] else 'no'} |"
    )
(RESULTS_DIR / "results_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

print("\n".join(markdown))