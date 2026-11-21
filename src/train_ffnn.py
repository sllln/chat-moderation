"""
Step 6b - Train the feedforward network with SGD and backpropagation (technique area A).

  - Input: message vectors built from our word2vec model (see neural_model.py)
  - Optimizer: stochastic gradient descent (SGD) with momentum
  - Loss: binary cross-entropy, with a higher weight for the rare "offensive" class
  - After every epoch we measure macro-F1 on the VALIDATION set and keep the
    best epoch ("early stopping").

Saves models/ffnn.pt
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from neural_model import FeedForwardNet, message_vector

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "ffnn.pt"

EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
PATIENCE = 10       # stop if validation macro-F1 has not improved for this many epochs

torch.manual_seed(42)

# 1) Turn every tweet into a 200-number vector.
word_vectors = Word2Vec.load(str(MODELS_DIR / "word2vec.model")).wv
train = pd.read_csv(DATA_DIR / "train.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")


def to_tensors(frame: pd.DataFrame):
    x = np.stack([message_vector(t, word_vectors) for t in frame["text"]])
    y = frame["label"].to_numpy(dtype=np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


x_train, y_train = to_tensors(train)
x_val, y_val = to_tensors(validation)
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# 2) Model, loss and optimizer.
model = FeedForwardNet(input_size=x_train.shape[1])
model.set_normalization(x_train)
n_clean = int((y_train == 0).sum())
n_offensive = int((y_train == 1).sum())
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_clean / n_offensive))
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# 3) Training loop.
print(f"{'epoch':>5} {'train loss':>11} {'validation macro-F1':>20}")
best_f1, best_epoch, epochs_without_improvement = -1.0, 0, 0
for epoch in range(1, EPOCHS + 1):
    model.train()                      # dropout ON
    total_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()          # forget the gradients of the previous batch
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()                # backpropagation: compute the gradients
        optimizer.step()               # SGD: move every weight a little against its gradient
        total_loss += loss.item() * len(x_batch)

    model.eval()                       # dropout OFF
    with torch.no_grad():
        predictions = (torch.sigmoid(model(x_val)) >= 0.5).int().numpy()
    f1 = f1_score(y_val.numpy(), predictions, average="macro")
    print(f"{epoch:>5} {total_loss / len(x_train):>11.4f} {f1:>20.3f}")

    if f1 > best_f1:
        best_f1, best_epoch, epochs_without_improvement = f1, epoch, 0
        torch.save(model.state_dict(), MODEL_FILE)
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs -> stopping early.")
            break

size_mb = MODEL_FILE.stat().st_size / 1024 / 1024
print(f"\nBest epoch: {best_epoch} (validation macro-F1 = {best_f1:.3f})")
print(f"Saved to {MODEL_FILE} ({size_mb:.2f} MB)")