"""Groft-til-fint generering med en trænet NCM.

    z_hat = 0
    for k in niveauer:                    # 8 skridt for en PRD på 256 ord
        logits = generator(idé, z_hat)    # alle koder i niveau k, parallelt
        koder  = sample(logits)
        z_hat += tokenizer.level_contribution(k, koder)
    ord = tokenizer.decode(z_hat).argmax()

Med `trajectory=True` afkoder vi også z_hat efter hvert niveau. Så kan man
se PRD'en komme i fokus: først en sløret essens, så skarpere detaljer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .concept_tokenizer import ConceptTokenizer
from .generator import NextConceptTransformer


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Vælg ét id pr. række af (..., V) logits. Temperatur 0 = vælg altid det mest sandsynlige."""
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
    tokens: torch.Tensor  # (B, target_len) ord-id'er for de færdige PRD'er
    nfe: int  # antal forward passes i generatoren (= antal niveauer)
    # trajectory[k] = ord-id'er afkodet efter niveau 0..k, så man kan kigge med
    trajectory: list[torch.Tensor] = field(default_factory=list)


@torch.no_grad()
def generate_ncm(
    tokenizer: ConceptTokenizer,
    generator: NextConceptTransformer,
    prompt: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    trajectory: bool = False,
) -> Generation:
    """Generer PRD'er for et batch af idéer (prompt: (B, prompt_len) ord-id'er)."""
    batch = prompt.shape[0]
    target_len = tokenizer.position.shape[0]
    z_hat = torch.zeros(batch, target_len, tokenizer.cfg.d_latent, device=prompt.device)
    steps = []

    for k in range(generator.n_levels):
        logits = generator.predict_level(prompt, k, z_hat)
        codes = sample_from_logits(logits, temperature, top_k)
        z_hat = z_hat + tokenizer.level_contribution(k, codes)
        if trajectory:
            steps.append(tokenizer.decode(z_hat).argmax(-1))

    tokens = tokenizer.decode(z_hat).argmax(-1)
    return Generation(tokens=tokens, nfe=generator.n_levels, trajectory=steps)
