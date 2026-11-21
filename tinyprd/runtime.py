"""Valg af enhed (GPU/CPU), seeds, datasæt som tensorer og checkpoints."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import torch

from .baseline import AutoregressiveBaseline
from .concept_tokenizer import ConceptTokenizer
from .config import (
    DATA_DIR,
    RUNS_DIR,
    BaselineConfig,
    ConceptTokenizerConfig,
    DataConfig,
    GeneratorConfig,
)
from .generator import NextConceptTransformer
from .prd import PrdRecord
from .synth import load_records
from .vocab import Vocab

TOKENIZER_CKPT = "concept_tokenizer.pt"
GENERATOR_CKPT = "generator.pt"
BASELINE_CKPT = "baseline.pt"


def pick_device() -> torch.device:
    """Apples GPU (MPS) hvis den findes, ellers CUDA, ellers CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---- data som tensorer -------------------------------------------------------


def load_vocab(data_dir: Path = DATA_DIR) -> Vocab:
    return Vocab.load(data_dir / "vocab.json")


def load_split(name: str, data_dir: Path = DATA_DIR) -> list[PrdRecord]:
    return load_records(data_dir / f"{name}.jsonl")


def encode_records(records: list[PrdRecord], vocab: Vocab, data_cfg: DataConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Poster -> (prompts (N, prompt_len), targets (N, target_len)) ord-id'er."""
    prompts = [vocab.encode(r.idea, data_cfg.prompt_len) for r in records]
    targets = [vocab.encode(r.render(), data_cfg.target_len, add_eos=True) for r in records]
    return torch.tensor(prompts), torch.tensor(targets)


def batches(n: int, batch_size: int, generator: torch.Generator):
    """Uendelig strøm af tilfældige batches af indekser."""
    while True:
        order = torch.randperm(n, generator=generator)
        for start in range(0, n - batch_size + 1, batch_size):
            yield order[start : start + batch_size]


# ---- checkpoints -----------------------------------------------------------


def save_checkpoint(path: Path, model: torch.nn.Module, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **metadata}, path)


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} findes ikke. Træn modellen først (se `tinyprd --help`).")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_tokenizer(device: torch.device, runs_dir: Path = RUNS_DIR) -> tuple[ConceptTokenizer, dict]:
    ckpt = _load(runs_dir / TOKENIZER_CKPT)
    cfg = ConceptTokenizerConfig(**ckpt["config"])
    cfg.level_lens = tuple(cfg.level_lens)
    model = ConceptTokenizer(cfg, ckpt["vocab_size"], ckpt["data"]["target_len"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def load_generator(device: torch.device, runs_dir: Path = RUNS_DIR) -> tuple[NextConceptTransformer, dict]:
    ckpt = _load(runs_dir / GENERATOR_CKPT)
    tok = ckpt["tokenizer"]
    model = NextConceptTransformer(
        GeneratorConfig(**ckpt["config"]),
        vocab_size=ckpt["vocab_size"],
        prompt_len=ckpt["data"]["prompt_len"],
        target_len=ckpt["data"]["target_len"],
        level_lens=tuple(tok["level_lens"]),
        d_latent=tok["d_latent"],
        codebook_size=tok["codebook_size"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def load_baseline(device: torch.device, runs_dir: Path = RUNS_DIR) -> tuple[AutoregressiveBaseline, dict]:
    ckpt = _load(runs_dir / BASELINE_CKPT)
    model = AutoregressiveBaseline(
        BaselineConfig(**ckpt["config"]),
        vocab_size=ckpt["vocab_size"],
        prompt_len=ckpt["data"]["prompt_len"],
        target_len=ckpt["data"]["target_len"],
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def config_dict(config) -> dict:
    return asdict(config)
