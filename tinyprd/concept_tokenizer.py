"""Trin 1: koncept-tokenizeren (en multiskala residual-VQ-VAE).

Den lærer at beskrive en PRD som et hierarki af koder, fra groft til fint,
og at lave sådan et hierarki om til ord igen.

    ord (256)  --encode-->  latente vektorer z (256 x d)
                                 |
            niveau 1:   2 koder  |  "hvad handler PRD'en groft sagt om?"
            niveau 2:   8 koder  |  ... lidt flere detaljer
              ...                |
            niveau 8: 256 koder  v  "præcis hvilket ord står hvor"
                                 |
    ord (256)  <--decode--  rekonstruktion z_hat (256 x d)

Residual-kvantisering, trin for trin, for niveau k:

    1. rest      = z - z_hat               det de grovere niveauer endnu mangler
    2. u_k       = downsample(rest)        pres det ned til niveauets længde
    3. koder_k   = nærmeste kode i kodebogen for hver vektor i u_k
    4. z_hat    += refine(upsample(kodebog[koder_k]))

Niveau 1 fanger altså essensen med 2 koder, og hvert senere niveau tilføjer
kun det, der stadig mangler. Det er det, der gør hierarkiet groft-til-fint.

Nested dropout (kun under træning): halvdelen af gangene får decoderen kun
de første par niveauer. Den skal stadig gendanne hele PRD'en, så de grove
niveauer tvinges til selv at bære den vigtigste information.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ConceptTokenizerConfig
from .layers import TransformerBlock, downsample_matrix, resample, upsample_matrix


class LevelRefiner(nn.Module):
    """Glatter et niveaus udstrakte bidrag med en lille foldning (convolution).

    Starter som identiteten (foldningen er nul fra start) og lærer en rettelse.
    Svarer til VAR-modellens "phi"-lag pr. skala.
    """

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


def nearest_code(vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Indeks for den nærmeste kode i kodebogen, for hver vektor.

    vectors: (B, N, D), codebook: (V, D) -> (B, N) indekser.
    Bruger |a - b|^2 = |a|^2 - 2 a.b + |b|^2, så vi undgår en kæmpe (B, N, V, D)-tensor.
    """
    distances = (
        vectors.pow(2).sum(-1, keepdim=True)
        - 2 * vectors @ codebook.t()
        + codebook.pow(2).sum(-1)
    )
    return distances.argmin(dim=-1)


class ConceptTokenizer(nn.Module):
    def __init__(self, cfg: ConceptTokenizerConfig, vocab_size: int, target_len: int):
        super().__init__()
        if cfg.level_lens[-1] != target_len:
            raise ValueError("det fineste niveau skal have én kode pr. ord (target_len)")
        self.cfg = cfg
        self.level_lens = tuple(cfg.level_lens)
        self.n_levels = len(self.level_lens)
        width = cfg.d_latent

        # Encoder: ord -> én latent vektor pr. ordposition.
        self.word_embed = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.randn(target_len, width) * 0.02)
        self.encoder = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(width)

        # Én kodebog og én refiner pr. niveau.
        self.codebooks = nn.ModuleList(nn.Embedding(cfg.codebook_size, width) for _ in self.level_lens)
        self.refiners = nn.ModuleList(LevelRefiner(width) for _ in self.level_lens)

        # Decoder: rekonstruktion z_hat -> ord-sandsynligheder på hver position.
        self.decoder = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(width)
        self.to_words = nn.Linear(width, vocab_size)

        # Faste matricer til at skifte længde mellem ord-gitteret og hvert niveau.
        for k, level_len in enumerate(self.level_lens):
            self.register_buffer(f"down_{k}", downsample_matrix(target_len, level_len), persistent=False)
            self.register_buffer(f"up_{k}", upsample_matrix(level_len, target_len), persistent=False)

        # Statistik over brugte koder, så ubrugte koder kan genbruges.
        self.register_buffer("code_usage", torch.zeros(self.n_levels, cfg.codebook_size), persistent=False)
        self._recent_inputs: list[torch.Tensor] = []

    # ---- de tre kerne-operationer -------------------------------------

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) ord-id'er -> (B, L, D) latente vektorer."""
        x = self.word_embed(tokens) + self.position
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def quantize(self, z: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Residual-kvantiser de latente vektorer, niveau for niveau.

        Returnerer:
            codes     - liste med K tensorer; codes[k] har formen (B, level_lens[k])
            partials  - liste med K rekonstruktioner; partials[k] bruger niveau 0..k
            vq_loss   - kodebogs- + commitment-loss, gennemsnit over niveauer
        """
        z_hat = torch.zeros_like(z)
        codes, partials = [], []
        vq_loss = z.new_zeros(())
        self._recent_inputs = []

        for k in range(self.n_levels):
            # Det der stadig mangler, i dette niveaus opløsning.
            u = resample(getattr(self, f"down_{k}"), z - z_hat)
            index = nearest_code(u, self.codebooks[k].weight)
            quantized = self.codebooks[k](index)

            # Kodebogs-loss flytter koderne mod input; commitment-loss flytter
            # input mod koderne (vægtet med beta).
            vq_loss = vq_loss + F.mse_loss(quantized, u.detach())
            vq_loss = vq_loss + self.cfg.commitment_beta * F.mse_loss(u, quantized.detach())

            # Straight-through: fremad bruges koden, men baglæns sendes
            # gradienten til u, som om kvantiseringen slet ikke var der.
            quantized = u + (quantized - u).detach()

            z_hat = z_hat + self.refiners[k](resample(getattr(self, f"up_{k}"), quantized))
            codes.append(index)
            partials.append(z_hat)

            if self.training:
                self.code_usage[k] += torch.bincount(index.flatten(), minlength=self.cfg.codebook_size)
                self._recent_inputs.append(u.detach().reshape(-1, u.shape[-1]))

        return codes, partials, vq_loss / self.n_levels

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        """(B, L, D) rekonstruktion -> (B, L, vocab) ord-logits."""
        x = z_hat + self.position
        for block in self.decoder:
            x = block(x)
        return self.to_words(self.decoder_norm(x))

    # ---- hjælpere som generatoren bruger ---------------------------------

    def level_contribution(self, k: int, codes: torch.Tensor) -> torch.Tensor:
        """Det niveau k lægger til rekonstruktionen, givet dets koder (B, len_k)."""
        return self.refiners[k](resample(getattr(self, f"up_{k}"), self.codebooks[k](codes)))

    def partials_from_codes(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Den samlede rekonstruktion efter hvert niveau, kun ud fra koderne."""
        z_hat, partials = None, []
        for k, level_codes in enumerate(codes):
            contribution = self.level_contribution(k, level_codes)
            z_hat = contribution if z_hat is None else z_hat + contribution
            partials.append(z_hat)
        return partials

    @torch.no_grad()
    def tokenize(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Ord -> hierarki af koder (det trin 2 lærer at generere)."""
        codes, _, _ = self.quantize(self.encode(tokens))
        return codes

    # ---- træning ------------------------------------------------------

    def forward(self, tokens: torch.Tensor, pad_id: int) -> dict[str, torch.Tensor]:
        """Ét træningsskridt: encode, kvantiser, decode (med nested dropout)."""
        z = self.encode(tokens)
        _, partials, vq_loss = self.quantize(z)

        # Nested dropout: for hvert eksempel afkodes fra alle niveauer, eller
        # (med sandsynlighed p) kun fra de første `cutoff` niveauer.
        batch = tokens.shape[0]
        cutoff = torch.full((batch,), self.n_levels - 1, device=tokens.device)
        if self.training:
            truncate = torch.rand(batch, device=tokens.device) < self.cfg.nested_dropout_p
            random_cutoff = torch.randint(0, self.n_levels, (batch,), device=tokens.device)
            cutoff = torch.where(truncate, random_cutoff, cutoff)
        z_hat = torch.stack(partials, dim=1)[torch.arange(batch, device=tokens.device), cutoff]

        logits = self.decode(z_hat)
        reconstruction = F.cross_entropy(logits.transpose(1, 2), tokens, ignore_index=pad_id)
        return {"loss": reconstruction + vq_loss, "reconstruction": reconstruction, "vq": vq_loss}

    @torch.no_grad()
    def reseed_dead_codes(self) -> float:
        """Erstat koder, som ingen har brugt siden sidst, med nylige input.

        Uden dette bliver mange koder aldrig valgt, og kodebogen "skrumper".
        Kaldt én gang i starten af træningen starter den også alle kodebøger
        ud fra rigtige data. Returnerer andelen af brugte koder (0 til 1).
        """
        used_fraction = (self.code_usage > 0).float().mean().item()
        for k, inputs in enumerate(self._recent_inputs):
            dead = (self.code_usage[k] == 0).nonzero().flatten()
            if len(dead) == 0:
                continue
            picks = torch.randint(0, inputs.shape[0], (len(dead),), device=inputs.device)
            noise = 0.01 * torch.randn(len(dead), inputs.shape[1], device=inputs.device)
            self.codebooks[k].weight.data[dead] = inputs[picks] + noise
        self.code_usage.zero_()
        return used_fraction
