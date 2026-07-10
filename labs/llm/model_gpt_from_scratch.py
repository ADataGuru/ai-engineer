import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import math

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C) = (batch_size, time=block_size, channels=vocab_size)

        if targets is None:  # don't compute loss if targets not given (used for generation)
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B x T, C)
            targets = targets.view(B * T)  # (B x T)
            loss = F.cross_entropy(logits, targets)  # F.cross_entropy inputs shape (B, C, T)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1) ... (B, T+max_new_tokens)
        return idx


def init_model(filename: str):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    unique_chars = sorted(list(set(text)))
    vocab_size = len(unique_chars)

    return text, unique_chars, vocab_size


def create_tokenizers(chars: list):
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # encoder-decoder
    encode = lambda x: [stoi[i] for i in x]
    decode = lambda x: ''.join([itos[i] for i in x])

    return encode, decode


def create_tensor(text, encode):
    data = torch.tensor(encode(text), dtype=torch.long, device=torch.device('cpu'))
    print(data.shape, data.dtype)
    n = int(0.9 * len(data))  # first 90% will be train, remaining 10% will be val
    train_data = data[:n]
    val_data = data[n:]
    return data, train_data, val_data


def get_batch(split):
    # generate a small batch (batch_size, block_size) of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # randomly sample a bunch of block_size length sequences
    ix = torch.randint(len(data) - block_size, (batch_size,))  # (batch_size, )
    # the sequence (stack each sequence of the batch indices to form a tensor)
    x = torch.stack([data[i:i + block_size] for i in ix])  # (batch_size, block_size)
    # the target (next character)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (batch_size, block_size)
    return x, y


def visualize_tensor(tensor, row_labels=None, col_labels=None):
    """
    Visualise un tensor PyTorch sous forme de DataFrame pandas.

    Args:
        tensor     : torch.Tensor (1D ou 2D)
        row_labels : liste de noms pour les lignes (optionnel)
        col_labels : liste de noms pour les colonnes (optionnel)
    """
    t = tensor.detach().cpu().numpy()

    # Si le tensor est 1D → on le reshape en 2D
    if t.ndim == 1:
        t = t.reshape(1, -1)

    # Si le tensor est > 2D → on affiche chaque "tranche"
    if t.ndim > 2:
        for i in range(t.shape[0]):
            print(f"\n📐 Slice [{i}] — shape {t[i].shape}")
            visualize_tensor(torch.tensor(t[i]), row_labels, col_labels)
        return

    n_rows, n_cols = t.shape

    # Labels automatiques si non fournis
    rows = row_labels if row_labels else [f"row{i}" for i in range(n_rows)]
    cols = col_labels if col_labels else [f"col{j}" for j in range(n_cols)]

    df = pd.DataFrame(t, index=rows, columns=cols)
    print(f"\n📐 Shape: {tensor.shape}")
    print(df)
    print()


if __name__ == '__main__':
    block_size = 8  # context length

    text, chars, vocab_size = init_model('input.txt')
    encode, decode = create_tokenizers(chars)
    data, train_data, val_data = create_tensor(text, encode)
    train_data[:block_size + 1]