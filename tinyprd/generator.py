"""Trin 2: next-concept transformeren.

Givet idéen (prompten) og alt det, der er besluttet indtil nu, forudsiger
den ALLE koder i det næste niveau på én gang.

    skridt 1:  idé + (intet endnu)          -> niveau 1 koder  (2 koder)
    skridt 2:  idé + rekonstruktion(1)      -> niveau 2 koder  (8 koder)
    ...
    skridt 8:  idé + rekonstruktion(1..7)   -> niveau 8 koder  (256 koder)

Otte forward passes skriver en PRD på 256 ord. En autoregressiv model skal
bruge ét forward pass pr. ord.

Hvad modellen ser for niveau k
    Den frosne koncept-tokenizer laver koderne fra niveau 1..k-1 om til en
    rekonstruktion z_hat (256 x d). Vi presser den ned til niveau k's længde
    og projicerer den til modellens bredde. Altså: "her er den grove PRD
    indtil nu, i din opløsning; tilføj nu det næste lag af detaljer".

Træn alle niveauer i ét pass
    Under træning kender vi de rigtige koder for alle niveauer. Derfor kan
    input til alle niveauer bygges på forhånd og sættes efter hinanden i én
    lang sekvens:

        [ idé | niveau 1 | niveau 2 | ... | niveau 8 ]

    En blok-diagonal attention-maske holder niveauerne adskilt: hver blok må
    kigge på idéen og på sig selv, intet andet. Hvert niveau har allerede de
    grovere niveauer med i sit input z_hat, så det behøver ikke mere.
    Den samme maske gør generering billig: skridt k kører kun [ idé | niveau k ].

Forstyrrelse af betingelsen (corruption)
    Under generering er de grove niveauer *samplet*, så de indeholder fejl.
    Derfor bytter vi under træning en del af betingelses-koderne ud med
    tilfældige (målene forbliver rene). Så lærer modellen at lave gode
    detaljer oven på grove niveauer, der ikke er perfekte.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import GeneratorConfig
from .layers import TransformerBlock, downsample_matrix, finest_positions, resample


class NextConceptTransformer(nn.Module):
    def __init__(
        self,
        cfg: GeneratorConfig,
        vocab_size: int,
        prompt_len: int,
        target_len: int,
        level_lens: tuple[int, ...],
        d_latent: int,
        codebook_size: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.prompt_len = prompt_len
        self.level_lens = tuple(level_lens)
        self.n_levels = len(level_lens)
        width = cfg.d_model

        # Idéen: almindelige ord-embeddings + positioner.
        self.prompt_embed = nn.Embedding(vocab_size, width)
        self.prompt_position = nn.Parameter(torch.randn(prompt_len, width) * 0.02)

        # Niveauerne: projiceret rekonstruktion + "hvilket niveau" + "hvor i PRD'en".
        self.latent_in = nn.Linear(d_latent, width)
        self.level_embed = nn.Embedding(self.n_levels, width)
        self.position = nn.Parameter(torch.randn(target_len, width) * 0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(width)
        # Hvert niveau har sin egen kodebog, så hvert får sit eget output-hoved.
        self.heads = nn.ModuleList(nn.Linear(width, codebook_size) for _ in level_lens)

        for k, level_len in enumerate(level_lens):
            self.register_buffer(f"down_{k}", downsample_matrix(target_len, level_len), persistent=False)
            self.register_buffer(f"pos_{k}", finest_positions(level_len, target_len), persistent=False)

    # ---- byg input-sekvensen -----------------------------------

    def embed_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        return self.prompt_embed(prompt) + self.prompt_position

    def embed_level(self, k: int, z_hat_before: torch.Tensor) -> torch.Tensor:
        """Input til niveau k: rekonstruktionen indtil nu, i niveau k's længde."""
        x = self.latent_in(resample(getattr(self, f"down_{k}"), z_hat_before))
        return x + self.level_embed.weight[k] + self.position[getattr(self, f"pos_{k}")]

    def attention_mask(self, segment_ids: torch.Tensor, prompt_is_real: torch.Tensor) -> torch.Tensor:
        """Blok-diagonal maske over [idé | blokke...].

        segment_ids: (T,) med 0 for idéens positioner og k+1 for niveau k.
        prompt_is_real: (B, prompt_len) False på idéens <pad>-positioner.
        Position i må kigge på position j, hvis j er et rigtigt ord i idéen,
        eller hvis i og j er i samme blok. Returnerer (B, 1, T, T).
        """
        same_block = segment_ids[:, None] == segment_ids[None, :]
        key_is_prompt = (segment_ids == 0)[None, :]
        allowed = same_block | key_is_prompt  # (T, T)

        batch = prompt_is_real.shape[0]
        n_levels_part = segment_ids.shape[0] - prompt_is_real.shape[1]
        key_is_real = torch.cat(
            [prompt_is_real, prompt_is_real.new_ones(batch, n_levels_part)], dim=1
        )  # (B, T)
        return (allowed[None] & key_is_real[:, None, :])[:, None]

    def run(self, prompt: torch.Tensor, pad_id: int, level_inputs: list[tuple[int, torch.Tensor]]):
        """Kør transformeren på [idé | de givne niveauer]; returner logits pr. niveau."""
        parts = [self.embed_prompt(prompt)]
        segments = [torch.zeros(self.prompt_len, dtype=torch.long)]
        for k, x in level_inputs:
            parts.append(x)
            segments.append(torch.full((x.shape[1],), k + 1, dtype=torch.long))
        x = torch.cat(parts, dim=1)
        segment_ids = torch.cat(segments).to(prompt.device)

        mask = self.attention_mask(segment_ids, prompt != pad_id)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        logits, start = [], self.prompt_len
        for k, level_x in level_inputs:
            end = start + level_x.shape[1]
            logits.append(self.heads[k](x[:, start:end]))
            start = end
        return logits

    # ---- træning: alle niveauer i ét pass --------------------------------

    def forward(
        self,
        prompt: torch.Tensor,
        pad_id: int,
        partials: list[torch.Tensor],
        target_codes: list[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-entropy over hver kode i hvert niveau.

        partials[k] er rekonstruktionen efter niveau 0..k, bygget af (måske
        forstyrrede) koder. Niveau k betinges på partials[k-1], og niveau 0
        på en rekonstruktion, der kun er nuller.
        """
        before = [torch.zeros_like(partials[0])] + partials[:-1]
        level_inputs = [(k, self.embed_level(k, before[k])) for k in range(self.n_levels)]
        logits = self.run(prompt, pad_id, level_inputs)

        # Hver kode tæller lige meget (artiklens alpha = 0 vægtning).
        all_logits = torch.cat(logits, dim=1)
        all_targets = torch.cat(target_codes, dim=1)
        return F.cross_entropy(all_logits.transpose(1, 2), all_targets)

    # ---- generering: ét niveau ad gangen ----------------------------------

    def predict_level(self, prompt: torch.Tensor, pad_id: int, k: int, z_hat_before: torch.Tensor) -> torch.Tensor:
        """Logits for alle koder i niveau k, i ét enkelt forward pass."""
        return self.run(prompt, pad_id, [(k, self.embed_level(k, z_hat_before))])[0]


def corrupt_codes(codes: list[torch.Tensor], codebook_size: int, start: float, end: float) -> list[torch.Tensor]:
    """Byt en andel af koderne ud med tilfældige: `start` på niveau 1, faldende til `end`."""
    n = len(codes)
    corrupted = []
    for k, level_codes in enumerate(codes):
        rate = start + (end - start) * (k / max(1, n - 1))
        replace = torch.rand(level_codes.shape, device=level_codes.device) < rate
        random_codes = torch.randint_like(level_codes, codebook_size)
        corrupted.append(torch.where(replace, random_codes, level_codes))
    return corrupted
