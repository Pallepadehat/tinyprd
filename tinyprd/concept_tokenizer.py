"""Stage 1: the concept tokenizer (a multiscale residual VQ-VAE).

It learns to describe a PRD as a coarse-to-fine hierarchy of discrete codes,
and to turn such a hierarchy back into words.

    words (256)  --encode-->  latents z (256 x d)
                                 |
            level 1:   2 codes   |  "what is this PRD about, roughly?"
            level 2:   8 codes   |  ... a bit more detail
              ...                |
            level 8: 256 codes   v  "exactly which word goes where"
                                 |
    words (256)  <--decode--  reconstruction z_hat (256 x d)

Residual quantization, step by step, for level k:

    1. residual  = z - z_hat               what the coarser levels still miss
    2. u_k       = downsample(residual)    squeeze it to this level's length
    3. codes_k   = nearest codebook entry for every vector in u_k
    4. z_hat    += refine(upsample(codebook[codes_k]))

So level 1 captures the gist with 2 codes, and every later level only adds
what is still missing. That is what makes the hierarchy coarse-to-fine.

Nested dropout (training only): half the time, the decoder only gets the
first few levels. It still has to reconstruct the whole PRD, so the coarse
levels are forced to carry the most important information on their own.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import ConceptTokenizerConfig
from .layers import TransformerBlock, downsample_matrix, resample, upsample_matrix


class LevelRefiner(nn.Module):
    """Smooths a level's upsampled contribution with a small convolution.

    Starts as the identity (zero-initialised conv), and learns a correction.
    Plays the role of VAR's per-scale "phi" layer.
    """

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, D)
        return x + self.conv(x.transpose(1, 2)).transpose(1, 2)


def nearest_code(vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Index of the closest codebook row for every vector.

    vectors: (B, N, D), codebook: (V, D) -> (B, N) indices.
    Uses |a - b|^2 = |a|^2 - 2 a.b + |b|^2 to avoid building a (B, N, V, D) tensor.
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
            raise ValueError("the finest level must have one code per word (target_len)")
        self.cfg = cfg
        self.level_lens = tuple(cfg.level_lens)
        self.n_levels = len(self.level_lens)
        width = cfg.d_latent

        # Encoder: words -> one latent vector per word position.
        self.word_embed = nn.Embedding(vocab_size, width)
        self.position = nn.Parameter(torch.randn(target_len, width) * 0.02)
        self.encoder = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(width)

        # One codebook and one refiner per level.
        self.codebooks = nn.ModuleList(nn.Embedding(cfg.codebook_size, width) for _ in self.level_lens)
        self.refiners = nn.ModuleList(LevelRefiner(width) for _ in self.level_lens)

        # Decoder: reconstruction z_hat -> word logits at every position.
        self.decoder = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_decoder_layers)
        )
        self.decoder_norm = nn.LayerNorm(width)
        self.to_words = nn.Linear(width, vocab_size)

        # Fixed resize matrices between the word grid and each level.
        for k, level_len in enumerate(self.level_lens):
            self.register_buffer(f"down_{k}", downsample_matrix(target_len, level_len), persistent=False)
            self.register_buffer(f"up_{k}", upsample_matrix(level_len, target_len), persistent=False)

        # Code usage statistics, used to re-seed codes that are never chosen.
        self.register_buffer("code_usage", torch.zeros(self.n_levels, cfg.codebook_size), persistent=False)
        self._recent_inputs: list[torch.Tensor] = []

    # ---- the three core operations -------------------------------------

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, L) word ids -> (B, L, D) latents."""
        x = self.word_embed(tokens) + self.position
        for block in self.encoder:
            x = block(x)
        return self.encoder_norm(x)

    def quantize(self, z: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """Residual-quantize latents level by level.

        Returns:
            codes     - list of K tensors, codes[k] has shape (B, level_lens[k])
            partials  - list of K reconstructions; partials[k] uses levels 0..k
            vq_loss   - codebook + commitment loss, averaged over levels
        """
        z_hat = torch.zeros_like(z)
        codes, partials = [], []
        vq_loss = z.new_zeros(())
        self._recent_inputs = []

        for k in range(self.n_levels):
            # What is still missing, at this level's resolution.
            u = resample(getattr(self, f"down_{k}"), z - z_hat)
            index = nearest_code(u, self.codebooks[k].weight)
            quantized = self.codebooks[k](index)

            # Codebook loss moves codes toward inputs; commitment loss moves
            # inputs toward codes (weighted by beta).
            vq_loss = vq_loss + F.mse_loss(quantized, u.detach())
            vq_loss = vq_loss + self.cfg.commitment_beta * F.mse_loss(u, quantized.detach())

            # Straight-through estimator: the forward pass uses the code, the
            # backward pass sends gradients to u as if quantization were identity.
            quantized = u + (quantized - u).detach()

            z_hat = z_hat + self.refiners[k](resample(getattr(self, f"up_{k}"), quantized))
            codes.append(index)
            partials.append(z_hat)

            if self.training:
                self.code_usage[k] += torch.bincount(index.flatten(), minlength=self.cfg.codebook_size)
                self._recent_inputs.append(u.detach().reshape(-1, u.shape[-1]))

        return codes, partials, vq_loss / self.n_levels

    def decode(self, z_hat: torch.Tensor) -> torch.Tensor:
        """(B, L, D) reconstruction -> (B, L, vocab) word logits."""
        x = z_hat + self.position
        for block in self.decoder:
            x = block(x)
        return self.to_words(self.decoder_norm(x))

    # ---- helpers used by the generator ---------------------------------

    def level_contribution(self, k: int, codes: torch.Tensor) -> torch.Tensor:
        """What level k adds to the reconstruction, given its codes (B, len_k)."""
        return self.refiners[k](resample(getattr(self, f"up_{k}"), self.codebooks[k](codes)))

    def partials_from_codes(self, codes: list[torch.Tensor]) -> list[torch.Tensor]:
        """Cumulative reconstructions after each level, from codes alone."""
        z_hat, partials = None, []
        for k, level_codes in enumerate(codes):
            contribution = self.level_contribution(k, level_codes)
            z_hat = contribution if z_hat is None else z_hat + contribution
            partials.append(z_hat)
        return partials

    @torch.no_grad()
    def tokenize(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """Words -> hierarchy of codes (what stage 2 learns to generate)."""
        codes, _, _ = self.quantize(self.encode(tokens))
        return codes

    # ---- training ------------------------------------------------------

    def forward(self, tokens: torch.Tensor, pad_id: int) -> dict[str, torch.Tensor]:
        """Training step: encode, quantize, decode (with nested dropout)."""
        z = self.encode(tokens)
        _, partials, vq_loss = self.quantize(z)

        # Nested dropout: per example, decode from all levels, or (with
        # probability p) only from the first `cutoff` levels.
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
        """Replace codes nobody used since the last call with recent inputs.

        Without this, many codes are never picked and the codebook shrinks.
        Called once at the start of training, it also initialises every
        codebook from real data. Returns the codebook utilization (0 to 1).
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
