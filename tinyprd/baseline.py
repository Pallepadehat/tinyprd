"""The comparison model: an ordinary autoregressive (GPT-style) transformer.

It reads the idea, then writes the PRD one word at a time:

    [ idea words ... <pad> | prd word 1, prd word 2, ... <eos> ]
                             ^ predict each word from everything before it

Same data, same vocabulary, similar size as the NCM generator, so the
comparison asks one question: does coarse-to-fine generation help?

For readability, generation re-runs the full sequence for every new word (no
KV cache). That makes the baseline slower than an optimised one, so compare
speed using `nfe` (forward passes) rather than wall-clock alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import BaselineConfig
from .layers import TransformerBlock
from .sampling import sample_from_logits


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
        """Next-word loss on the PRD part only (the idea is given, not predicted)."""
        sequence = torch.cat([prompt, target], dim=1)
        logits = self.logits(sequence[:, :-1])
        labels = sequence[:, 1:].clone()
        labels[:, : self.prompt_len - 1] = pad_id  # don't score predictions of the idea
        return F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=pad_id)

    @torch.no_grad()
    def generate(
        self, prompt: torch.Tensor, eos_id: int, temperature: float = 0.0, top_k: int = 0
    ) -> tuple[torch.Tensor, int]:
        """Write PRDs word by word. Returns (target ids, forward passes used)."""
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

