"""Coarse-to-fine generation with a trained NCM.

    z_hat = 0
    for k in levels:                      # 8 steps for a 256-word PRD
        logits = generator(idea, z_hat)   # all codes of level k, in parallel
        codes  = sample(logits)
        z_hat += tokenizer.level_contribution(k, codes)
    words = tokenizer.decode(z_hat).argmax()

With `trajectory=True` we also decode z_hat after every level, which shows
the PRD coming into focus: first a blurry gist, then sharper detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .concept_tokenizer import ConceptTokenizer
from .generator import NextConceptTransformer


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Pick one id per row of (..., V) logits. Temperature 0 means greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k > 0:
        kth_best = logits.topk(top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth_best, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
    return flat.reshape(probs.shape[:-1])


@dataclass
class Generation:
    tokens: torch.Tensor  # (B, target_len) word ids of the final PRDs
    nfe: int  # generator forward passes used (= number of levels)
    # trajectory[k] = word ids decoded after levels 0..k, for inspection
    trajectory: list[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def generate_ncm(
    tokenizer: ConceptTokenizer,
    generator: NextConceptTransformer,
    prompt: torch.Tensor,
    pad_id: int,
    temperature: float = 0.0,
    top_k: int = 0,
    trajectory: bool = False,
) -> Generation:
    """Generate PRDs for a batch of ideas (prompt: (B, prompt_len) word ids)."""
    batch = prompt.shape[0]
    target_len = tokenizer.position.shape[0]
    z_hat = torch.zeros(batch, target_len, tokenizer.cfg.d_latent, device=prompt.device)
    steps = []

    for k in range(generator.n_levels):
        logits = generator.predict_level(prompt, pad_id, k, z_hat)
        codes = sample_from_logits(logits, temperature, top_k)
        z_hat = z_hat + tokenizer.level_contribution(k, codes)
        if trajectory:
            steps.append(tokenizer.decode(z_hat).argmax(-1))

    tokens = tokenizer.decode(z_hat).argmax(-1)
    return Generation(tokens=tokens, nfe=generator.n_levels, trajectory=steps)
