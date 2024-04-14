import math

import torch
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size: int, head_size: int, dropout: float):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        res = (q @ k.transpose(-2, -1)) // math.sqrt(self.head_size)
        return F.softmax(res, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        heads_count: int,
        head_size: int,
        embed_size: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            ScaledDotProductAttention(embed_size, head_size, dropout)
            for _ in range(heads_count)
        )
        self.projection = nn.Linear(heads_count * head_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        concatenated = torch.cat([head(x) for head in self.heads], dim=-1)
        res = self.projection(concatenated)
        return self.dropout(res)


class FeedForward(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.feedforward(x)


class EncoderBlock(nn.Module):
    def __init__(self, heads_count: int, embed_size: int, dropout: float):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(
            heads_count, embed_size // heads_count, embed_size, dropout
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.feedforward = FeedForward(embed_size, dropout)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        multihead_attention_res = self.multihead_attention(x)
        multihead_attention_res = self.norm1(multihead_attention_res + x)

        feedforward_res = self.feedforward(multihead_attention_res)
        feedforward_res = self.norm2(feedforward_res + multihead_attention_res)
        return feedforward_res


class PositionEncoding(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, x: torch.tensor) -> torch.tensor:
        sequence = x[-1]
        indices = torch.arange(0, self.embed_size, dtype=torch.float)
        indices[1::2] -= 1
        positions = torch.arange(0, len(sequence), dtype=torch.float)

        indices.div_(self.embed_size)
        indices = torch.pow(10000, indices)

        indices.unsqueeze_(0)
        positions.unsqueeze_(1)

        pe = positions / indices
        pe[:, ::2] = torch.sin(pe[:, ::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])

        return torch.stack([pe for _ in range(x.size(0))])


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_size: int,
        embed_size: int = 512,
        heads_count: int = 8,
        blocks_count: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    heads_count=heads_count,
                    embed_size=embed_size,
                    dropout=dropout,
                )
                for _ in range(blocks_count)
            ]
        )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional = PositionEncoding(embed_size)

    def forward(self, x):
        return self.blocks(self.embedding(x) + self.positional(x))


if __name__ == "__main__":
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    seq = text.split()
    words = set(seq)
    vocab = dict()
    for i, word in enumerate(words):
        vocab[word] = i
    tokens = [vocab[t] for t in seq]
    batch_len = len(tokens) // 3
    batches = [
        tokens[: batch_len],
        tokens[batch_len : 2 * batch_len],
        tokens[2 * batch_len : 3 * batch_len],
    ]

    encoder = Encoder(len(vocab), len(batches[0]))
    print(encoder.forward(torch.tensor(batches)))
