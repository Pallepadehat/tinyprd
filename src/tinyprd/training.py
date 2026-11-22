"""Træning af de tre modeller.

Rækkefølgen er vigtig:

    1. train_tokenizer   lærer at lave tekst om til et hierarki af koder (og tilbage)
    2. train_generator   lærer at forudsige koderne niveau for niveau (kræver 1)
    3. train_baseline    en almindelig ord-for-ord model, til sammenligning

Alle tre bruger den samme simple løkke: tag et tilfældigt batch, beregn
loss, tag et skridt med AdamW, og log undervejs.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .config import RUNS_DIR, Presets, to_dict
from .models.baseline import AutoregressiveBaseline
from .models.concept_tokenizer import ConceptTokenizer
from .models.generator import NextConceptTransformer, corrupt_codes
from .runtime import (
    BASELINE_CKPT,
    GENERATOR_CKPT,
    TOKENIZER_CKPT,
    batches,
    count_parameters,
    encode_records,
    load_split,
    load_tokenizer,
    load_vocab,
    pick_device,
    save_checkpoint,
    seed_everything,
)


def learning_rate(step: int, total: int, peak: float, warmup: int) -> float:
    """Lineær opvarmning, derefter cosinus-fald ned til 10 % af toppen."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


class Logger:
    """Skriver loss med jævne mellemrum, så man kan følge med i terminalen."""

    def __init__(self, name: str, total: int, every: int = 100):
        self.name, self.total, self.every = name, total, every
        self.start = time.time()

    def __call__(self, step: int, **values: float) -> None:
        if step % self.every and step != self.total - 1:
            return
        parts = "  ".join(f"{key} {value:.4f}" for key, value in values.items())
        print(f"[{self.name}] step {step + 1:>5}/{self.total}  {parts}  ({time.time() - self.start:.0f}s)", flush=True)


# --------------------------------------------------------------------------
# 1. Koncept-tokenizer
# --------------------------------------------------------------------------


def train_tokenizer(preset: Presets, runs_dir: Path = RUNS_DIR) -> ConceptTokenizer:
    cfg, data_cfg = preset.tokenizer, preset.data
    seed_everything(data_cfg.seed)
    device = pick_device()
    vocab = load_vocab()
    _, targets = encode_records(load_split("train"), vocab, data_cfg)
    targets = targets.to(device)

    model = ConceptTokenizer(cfg, len(vocab), data_cfg.target_len).to(device)
    print(f"Koncept-tokenizer: {count_parameters(model) / 1e6:.1f}M parametre, enhed: {device}")
    optimizer = make_optimizer(model, cfg.lr)
    log = Logger("tokenizer", cfg.steps)
    index_stream = batches(len(targets), cfg.batch_size, torch.Generator().manual_seed(data_cfg.seed))

    model.train()
    for step in range(cfg.steps):
        set_lr(optimizer, learning_rate(step, cfg.steps, cfg.lr, cfg.warmup_steps))
        losses = model(targets[next(index_stream).to(device)], vocab.pad_id)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Første skridt: start alle kodebøger fra rigtige data.
        # Derefter: genbrug koder, som ingen har valgt i et stykke tid.
        utilization = None
        if step == 0 or (step + 1) % cfg.dead_code_every == 0:
            utilization = model.reseed_dead_codes()
        log(step, loss=losses["loss"].item(), recon=losses["reconstruction"].item(), vq=losses["vq"].item(),
            **({"brugte_koder": utilization} if utilization is not None and step > 0 else {}))

    model.eval()
    save_checkpoint(
        runs_dir / TOKENIZER_CKPT, model,
        config=to_dict(cfg), data=to_dict(data_cfg), vocab_size=len(vocab),
    )
    print(f"Gemt: {runs_dir / TOKENIZER_CKPT}")
    return model


@torch.no_grad()
def tokenizer_accuracy(model: ConceptTokenizer, targets: torch.Tensor, pad_id: int, batch_size: int = 128) -> list[float]:
    """Andel ord der gendannes korrekt, når man afkoder fra de første k niveauer.

    Det sidste tal er "fuld rekonstruktion". De første tal viser, hvor meget
    af teksten de grove niveauer alene kan bære.
    """
    correct = torch.zeros(model.n_levels)
    total = 0
    for start in range(0, len(targets), batch_size):
        tokens = targets[start : start + batch_size]
        _, partials, _ = model.quantize(model.encode(tokens))
        real = tokens != pad_id
        for k, z_hat in enumerate(partials):
            predicted = model.decode(z_hat).argmax(-1)
            correct[k] += ((predicted == tokens) & real).sum().item()
        total += real.sum().item()
    return (correct / total).tolist()


# --------------------------------------------------------------------------
# 2. Next-concept generator
# --------------------------------------------------------------------------


@torch.no_grad()
def precompute_codes(tokenizer: ConceptTokenizer, targets: torch.Tensor, batch_size: int = 256) -> list[torch.Tensor]:
    """Lav alle PRD'er om til koder én gang. Tokenizeren er frosset, så de ændrer sig ikke."""
    per_level: list[list[torch.Tensor]] = [[] for _ in range(tokenizer.n_levels)]
    for start in range(0, len(targets), batch_size):
        for k, codes in enumerate(tokenizer.tokenize(targets[start : start + batch_size])):
            per_level[k].append(codes)
    return [torch.cat(level) for level in per_level]


def train_generator(preset: Presets, runs_dir: Path = RUNS_DIR) -> NextConceptTransformer:
    cfg, data_cfg = preset.generator, preset.data
    seed_everything(data_cfg.seed)
    device = pick_device()
    vocab = load_vocab()
    tokenizer, tok_ckpt = load_tokenizer(device, runs_dir)
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    prompts, targets = encode_records(load_split("train"), vocab, data_cfg)
    prompts, targets = prompts.to(device), targets.to(device)
    all_codes = precompute_codes(tokenizer, targets)
    tok_cfg = tokenizer.cfg

    model = NextConceptTransformer(
        cfg, len(vocab), data_cfg.prompt_len, data_cfg.target_len,
        tok_cfg.level_lens, tok_cfg.d_latent, tok_cfg.codebook_size,
    ).to(device)
    print(f"Next-concept generator: {count_parameters(model) / 1e6:.1f}M parametre, enhed: {device}")
    optimizer = make_optimizer(model, cfg.lr)
    log = Logger("generator", cfg.steps)
    index_stream = batches(len(prompts), cfg.batch_size, torch.Generator().manual_seed(data_cfg.seed))

    model.train()
    for step in range(cfg.steps):
        set_lr(optimizer, learning_rate(step, cfg.steps, cfg.lr, cfg.warmup_steps))
        idx = next(index_stream).to(device)
        clean = [level[idx] for level in all_codes]

        # Betingelsen bygges af *forstyrrede* koder; målet er de rene koder.
        noisy = corrupt_codes(clean, tok_cfg.codebook_size, cfg.corruption_start, cfg.corruption_end)
        with torch.no_grad():
            partials = tokenizer.partials_from_codes(noisy)

        loss = model(prompts[idx], partials, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log(step, loss=loss.item())

    model.eval()
    save_checkpoint(
        runs_dir / GENERATOR_CKPT, model,
        config=to_dict(cfg), data=to_dict(data_cfg), vocab_size=len(vocab), tokenizer=tok_ckpt["config"],
    )
    print(f"Gemt: {runs_dir / GENERATOR_CKPT}")
    return model


# --------------------------------------------------------------------------
# 3. Autoregressiv baseline
# --------------------------------------------------------------------------


def train_baseline(preset: Presets, runs_dir: Path = RUNS_DIR) -> AutoregressiveBaseline:
    cfg, data_cfg = preset.baseline, preset.data
    seed_everything(data_cfg.seed)
    device = pick_device()
    vocab = load_vocab()
    prompts, targets = encode_records(load_split("train"), vocab, data_cfg)
    prompts, targets = prompts.to(device), targets.to(device)

    model = AutoregressiveBaseline(cfg, len(vocab), data_cfg.prompt_len, data_cfg.target_len).to(device)
    print(f"Autoregressiv baseline: {count_parameters(model) / 1e6:.1f}M parametre, enhed: {device}")
    optimizer = make_optimizer(model, cfg.lr)
    log = Logger("baseline", cfg.steps)
    index_stream = batches(len(prompts), cfg.batch_size, torch.Generator().manual_seed(data_cfg.seed))

    model.train()
    for step in range(cfg.steps):
        set_lr(optimizer, learning_rate(step, cfg.steps, cfg.lr, cfg.warmup_steps))
        idx = next(index_stream).to(device)
        loss = model(prompts[idx], targets[idx], vocab.pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log(step, loss=loss.item())

    model.eval()
    save_checkpoint(
        runs_dir / BASELINE_CKPT, model,
        config=to_dict(cfg), data=to_dict(data_cfg), vocab_size=len(vocab),
    )
    print(f"Gemt: {runs_dir / BASELINE_CKPT}")
    return model
