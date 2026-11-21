"""
Our own neural network (technique area A) on top of our own word2vec vectors.

A message is turned into ONE fixed-size vector:
  - the average of its word vectors (the overall meaning), and
  - the maximum of each dimension (the strongest single signal, e.g. one swear word),
put side by side -> 2 x 100 = 200 numbers.

A small feedforward network then decides clean (0) / offensive (1).
Before the first layer, every input number is standardized (mean 0, std 1)
with statistics computed on the training data. They are stored inside the
network, so they are saved and loaded together with the weights.
"""
import numpy as np
import torch
from torch import nn

from text_cleaning import clean_full, tokenize


def message_vector(text: str, word_vectors) -> np.ndarray:
    """Average + max of the word2vec vectors of the words in the message."""
    words = [w for w in tokenize(clean_full(text)) if w in word_vectors]
    size = word_vectors.vector_size
    if not words:
        return np.zeros(2 * size, dtype=np.float32)
    vectors = np.stack([word_vectors[w] for w in words])
    return np.concatenate([vectors.mean(axis=0), vectors.max(axis=0)]).astype(np.float32)


class FeedForwardNet(nn.Module):
    """standardize -> input (200) -> hidden layer (128, ReLU) -> dropout -> output (1 score)"""

    def __init__(self, input_size: int = 200, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        # "Buffers" are saved with the model but are not trained.
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("std", torch.ones(input_size))
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def set_normalization(self, x_train: torch.Tensor) -> None:
        """Remember the mean and standard deviation of every input number."""
        self.mean = x_train.mean(dim=0)
        self.std = x_train.std(dim=0) + 1e-6   # + tiny number: never divide by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        # Returns a raw score ("logit"); sigmoid(score) = probability of "offensive".
        return self.layers(x).squeeze(-1)