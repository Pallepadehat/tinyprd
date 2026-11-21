"""Stage 2: the next-concept transformer.

Given the idea (the prompt) and everything decided so far, predict ALL codes
of the next level at once.

    step 1:  idea + (nothing yet)        -> level 1 codes  (2 codes)
    step 2:  idea + reconstruction(1)    -> level 2 codes  (8 codes)
    ...
    step 8:  idea + reconstruction(1..7) -> level 8 codes  (256 codes)

Eight forward passes write a 256-word PRD. An autoregressive model needs one
forward pass per word.

What the model sees for level k
    The frozen concept tokenizer turns the codes of levels 1..k-1 into a
    reconstruction z_hat (256 x d). We downsample it to level k's length and
    project it to the model width. "Here is the rough PRD so far, at your
    resolution; now add the next layer of detail."

Training all levels in one pass
    During training we know every level's true codes, so the inputs for all
    levels can be built up front and concatenated into one long sequence:

        [ idea | level 1 | level 2 | ... | level 8 ]

    A block-diagonal attention mask keeps levels apart: every block may look
    at the idea and at itself, nothing else. Each level already carries the
    coarser levels inside its input z_hat, so it needs no other context.
    The same mask makes inference cheap: step k only runs [ idea | level k ].

Conditioning corruption
    At inference the coarse levels are *sampled*, so they contain mistakes.
    During training we therefore replace a fraction of the conditioning codes
    with random ones (the targets stay clean), so the model learns to produce
    good detail on top of imperfect coarse levels.
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

        # The idea: ordinary word embeddings + positions.
        self.prompt_embed = nn.Embedding(vocab_size, width)
        self.prompt_position = nn.Parameter(torch.randn(prompt_len, width) * 0.02)

        # The levels: projected reconstruction + "which level" + "where in the PRD".
        self.latent_in = nn.Linear(d_latent, width)
        self.level_embed = nn.Embedding(self.n_levels, width)
        self.position = nn.Parameter(torch.randn(target_len, width) * 0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(width, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(width)
        # Each level has its own codebook, so each gets its own output head.
        self.heads = nn.ModuleList(nn.Linear(width, codebook_size) for _ in level_lens)

        for k, level_len in enumerate(level_lens):
            self.register_buffer(f"down_{k}", downsample_matrix(target_len, level_len), persistent=False)
            self.register_buffer(f"pos_{k}", finest_positions(level_len, target_len), persistent=False)

    # ---- building the input sequence -----------------------------------

    def embed_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        return self.prompt_embed(prompt) + self.prompt_position

    def embed_level(self, k: int, z_hat_before: torch.Tensor) -> torch.Tensor:
        """Input for level k: the reconstruction so far, at level k's length."""
        x = self.latent_in(resample(getattr(self, f"down_{k}"), z_hat_before))
        return x + self.level_embed.weight[k] + self.position[getattr(self, f"pos_{k}")]

    def attention_mask(self, segment_ids: torch.Tensor, prompt_is_real: torch.Tensor) -> torch.Tensor:
        """Block-diagonal mask over [idea | blocks...].

        segment_ids: (T,) with 0 for idea positions and k+1 for level k.
        prompt_is_real: (B, prompt_len) False at <pad> positions of the idea.
        Query i may attend to key j if j is a real idea word, or i and j are
        in the same block. Returns (B, 1, T, T).
        """
        same_block = segment_ids[:, None] == segment_ids[None, :]
        key_is_prompt = (segment_ids == 0)[None, :]
        allowed = same_block | key_is_prompt  # (T, T)

        batch = prompt_is_real.shape[0]
        n_levels_part = segment_ids.shape[0] - self.prompt_len
        key_is_real = torch.cat(
            [prompt_is_real, prompt_is_real.new_ones(batch, n_levels_part)], dim=1
        )  # (B, T)
        return (allowed[None] & key_is_real[:, None, :])[:, None]

    def run(self, prompt: torch.Tensor, pad_id: int, level_inputs: list[tuple[int, torch.Tensor]]):
        """Run the transformer on [idea | given levels]; return logits per level."""
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

    # ---- training: all levels in one pass --------------------------------

    def forward(
        self,
        prompt: torch.Tensor,
        pad_id: int,
        partials: list[torch.Tensor],
        target_codes: list[torch.Tensor],
    ) -> torch.Tensor:
        """Cross-entropy over every code of every level.

        partials[k] is the reconstruction after levels 0..k, built from
        (possibly corrupted) codes. Level k is conditioned on partials[k-1],
        and level 0 on an all-zero reconstruction.
        """
        before = [torch.zeros_like(partials[0])] + partials[:-1]
        level_inputs = [(k, self.embed_level(k, before[k])) for k in range(self.n_levels)]
        logits = self.run(prompt, pad_id, level_inputs)

        # Every code counts equally (the paper's alpha = 0 weighting).
        all_logits = torch.cat(logits, dim=1)
        all_targets = torch.cat(target_codes, dim=1)
        return F.cross_entropy(all_logits.transpose(1, 2), all_targets)

    # ---- inference: one level at a time ----------------------------------

    def predict_level(self, prompt: torch.Tensor, pad_id: int, k: int, z_hat_before: torch.Tensor) -> torch.Tensor:
        """Logits for every code of level k, in a single forward pass."""
        return self.run(prompt, pad_id, [(k, self.embed_level(k, z_hat_before))])[0]


def corrupt_codes(codes: list[torch.Tensor], codebook_size: int, start: float, end: float) -> list[torch.Tensor]:
    """Replace a fraction of codes with random ones, `start` at level 1 down to `end`."""
    n = len(codes)
    corrupted = []
    for k, level_codes in enumerate(codes):
        rate = start + (end - start) * (k / max(1, n - 1))
        replace = torch.rand(level_codes.shape, device=level_codes.device) < rate
        random_codes = torch.randint_like(level_codes, codebook_size)
        corrupted.append(torch.where(replace, random_codes, level_codes))
    return corrupted
