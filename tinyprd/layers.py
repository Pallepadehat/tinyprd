"""Fælles byggeklodser: en transformer-blok og ændring af sekvenslængde.

Alt her er standard. Det holdes lille, så den interessante kode
(koncept-tokenizeren og next-concept transformeren) er let at læse.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------
# Transformer-blok
# --------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """Multi-head self-attention med en valgfri sand/falsk-maske.

    mask: True betyder "må kigge på". Form (T, T) eller (B, 1, T, T).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model skal kunne deles med n_heads"
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = x.shape
        head_dim = width // self.n_heads
        # (B, T, 3*D) -> tre tensorer med formen (B, heads, T, head_dim)
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class TransformerBlock(nn.Module):
    """Pre-norm blok: x + attention(norm(x)), derefter x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


# --------------------------------------------------------------------------
# Skift af længde mellem niveauer
# --------------------------------------------------------------------------
#
# Hierarkiet skal flytte information mellem niveauer af forskellig længde,
# fx presse 256 ord-vektorer ned i 8 "koncept"-pladser, eller strække 8
# pladser ud over 256 positioner igen. Begge dele er lineære, så vi skriver
# dem som faste matricer:  ny = M @ sekvens,  hvor M har formen (dst_len, src_len).
#
# Eksplicitte matricer (i stedet for adaptive pooling) virker på alle
# enheder, også Apples MPS, og gør det nemt at se, hvad der sker.


def downsample_matrix(src_len: int, dst_len: int) -> torch.Tensor:
    """Gennemsnit: hver ny plads er gennemsnittet af de input, den dækker."""
    matrix = torch.zeros(dst_len, src_len)
    scale = src_len / dst_len
    for i in range(dst_len):
        start, end = i * scale, (i + 1) * scale
        for j in range(math.floor(start), math.ceil(end)):
            overlap = min(end, j + 1) - max(start, j)
            if overlap > 0:
                matrix[i, j] = overlap
    return matrix / matrix.sum(dim=1, keepdim=True)


def upsample_matrix(src_len: int, dst_len: int) -> torch.Tensor:
    """Lineær interpolation: hver ny plads blander sine to nærmeste input."""
    matrix = torch.zeros(dst_len, src_len)
    for i in range(dst_len):
        x = (i + 0.5) * src_len / dst_len - 0.5  # pladsens midte, målt i input-koordinater
        x = min(max(x, 0.0), src_len - 1)
        left = math.floor(x)
        right = min(left + 1, src_len - 1)
        weight = x - left
        matrix[i, left] += 1 - weight
        matrix[i, right] += weight
    return matrix


def resample(matrix: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
    """Brug en (dst, src)-matrix på en sekvens med formen (batch, src, bredde)."""
    return torch.matmul(matrix, sequence)


def finest_positions(level_len: int, target_len: int) -> torch.Tensor:
    """Hvor hver plads i et niveau ligger på det fineste gitter.

    Et niveau med 4 pladser over 256 ord har pladserne ved ord 32, 96, 160, 224.
    Én fælles positionstabel for alle niveauer fortæller modellen, at fx
    plads 0 i et groft niveau og de første ord i det fineste niveau beskriver
    den samme del af PRD'en.
    """
    centres = (torch.arange(level_len) + 0.5) * target_len / level_len - 0.5
    return centres.round().long().clamp(0, target_len - 1)
