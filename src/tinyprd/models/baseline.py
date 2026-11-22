"""Sammenligningsmodellen: en almindelig autoregressiv (GPT-agtig) transformer.

Den læser idéen og skriver så PRD'en ét ord ad gangen:

    [ idéens ord ... <pad> | prd ord 1, prd ord 2, ... <eos> ]
                             ^ forudsig hvert ord ud fra alt før det

Samme data, samme ordforråd og omtrent samme størrelse som NCM-generatoren.
Sammenligningen stiller derfor ét spørgsmål: hjælper groft-til-fint generering?

For at holde koden læsbar kører generering hele sekvensen igen for hvert nyt
ord (ingen KV-cache). Baseline er derfor langsommere end en optimeret model,
så sammenlign hastighed med `nfe` (antal forward passes), ikke kun med tid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import BaselineConfig
from ..sampling import sample_from_logits
from .layers import TransformerBlock


class AutoregressiveBaseline(nn.Module):
    def __init__(self, cfg: BaselineConfig, vocab_size: int, prompt_len: int, target_len: int):
        super().__init__()
        self.cfg = cfg
        self.prompt_len = prompt_len
        self.max_len = prompt_len + target_len
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.position = nn.Parameter(torch.randn(self.max_len, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size)
        self.register_buffer("causal", torch.tril(torch.ones(self.max_len, self.max_len, dtype=torch.bool)), persistent=False)

    def logits(self, sequence: torch.Tensor) -> torch.Tensor:
        length = sequence.shape[1]
        x = self.embed(sequence) + self.position[:length]
        mask = self.causal[:length, :length]
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))

    def forward(self, prompt: torch.Tensor, target: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Næste-ord-loss kun på PRD-delen (idéen er givet og skal ikke forudsiges)."""
        sequence = torch.cat([prompt, target], dim=1)
        logits = self.logits(sequence[:, :-1])
        labels = sequence[:, 1:].clone()
        labels[:, : self.prompt_len - 1] = pad_id  # idéens ord tæller ikke med i loss
        return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=pad_id)

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, eos_id: int, temperature: float = 0.0, top_k: int = 0
    ) -> tuple[torch.Tensor, int]:
        """Skriv PRD'er ord for ord. Returnerer (ord-id'er, antal forward passes)."""
        sequence = prompt
        finished = torch.zeros(prompt.shape[0], dtype=torch.bool, device=prompt.device)
        nfe = 0
        while sequence.shape[1] < self.max_len and not finished.all():
            next_logits = self.logits(sequence)[:, -1]
            nfe += 1
            next_word = sample_from_logits(next_logits, temperature, top_k)
            next_word = torch.where(finished, torch.full_like(next_word, eos_id), next_word)
            finished |= next_word == eos_id
            sequence = torch.cat([sequence, next_word[:, None]], dim=1)
        return sequence[:, self.prompt_len :], nfe

