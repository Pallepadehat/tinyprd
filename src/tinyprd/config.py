"""Alle hyperparametre samlet ét sted.

Hvert trin har sin egen dataclass. Konfigurationen gemmes i checkpoints, så en
trænet model altid ved, hvilke størrelser den blev bygget med.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/tinyprd/config.py → projektets rod
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass
class DataConfig:
    prompt_len: int = 48  # max antal ord i idéen (det modellen læser først)
    target_len: int = 256  # max antal ord i en PRD = størrelsen af det fineste niveau
    examples_per_combo: int = 40  # træningseksempler pr. (domæne, mønster)-par
    seed: int = 0


@dataclass
class ConceptTokenizerConfig:
    """Trin 1: den multiskala residual-VQ-VAE ("koncept-tokenizeren")."""

    # Længden af hvert niveau, groft -> fint. Det sidste skal være target_len.
    # Niveau 1 har 2 koder for hele PRD'en; niveau 8 har én kode pr. ord.
    level_lens: tuple[int, ...] = (2, 8, 16, 32, 64, 128, 192, 256)
    codebook_size: int = 512  # koder pr. niveau (hvert niveau har sin egen kodebog)
    d_latent: int = 128  # bredden af de latente vektorer
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    commitment_beta: float = 0.25  # hvor hårdt encoderen trækkes mod koderne
    nested_dropout_p: float = 0.5  # chance for at afkode fra et afkortet hierarki
    dead_code_every: int = 200  # skridt mellem genbrug af ubrugte koder

    # træning
    steps: int = 4000
    batch_size: int = 64
    lr: float = 1e-3
    warmup_steps: int = 200


@dataclass
class GeneratorConfig:
    """Trin 2: next-concept transformeren."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    # Andel af betingelses-koderne, der byttes ud med tilfældige under træning.
    # Falder lineært fra det groveste til det fineste niveau. Det lærer
    # modellen at komme sig over sine egne fejl under generering.
    corruption_start: float = 0.5
    corruption_end: float = 0.1

    # træning (generator og baseline ser præcis lige mange eksempler)
    steps: int = 3000
    batch_size: int = 32
    lr: float = 5e-4
    warmup_steps: int = 200


@dataclass
class BaselineConfig:
    """En almindelig autoregressiv transformer i samme størrelse, til sammenligning."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    steps: int = 3000
    batch_size: int = 32
    lr: float = 5e-4
    warmup_steps: int = 200


@dataclass
class Presets:
    """Navngivne størrelser. `tiny` er hurtig at træne; `base` er den større version."""

    name: str = "tiny"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: ConceptTokenizerConfig = field(default_factory=ConceptTokenizerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


def get_preset(name: str) -> Presets:
    preset = Presets(name=name)
    if name == "tiny":
        return preset
    if name == "base":
        preset.generator = GeneratorConfig(d_model=512, n_layers=8, n_heads=8, steps=6000)
        preset.baseline = BaselineConfig(d_model=512, n_layers=8, n_heads=8, steps=6000)
        return preset
    if name == "smoke":  # få skridt, kun for at tjekke at koden kører hele vejen
        preset.data.examples_per_combo = 4
        preset.tokenizer.steps = 20
        preset.tokenizer.batch_size = 16
        preset.generator = GeneratorConfig(d_model=64, n_layers=2, n_heads=4, steps=20, batch_size=16)
        preset.baseline = BaselineConfig(d_model=64, n_layers=2, n_heads=4, steps=20, batch_size=16)
        return preset
    raise ValueError(f"ukendt preset {name!r}; vælg tiny, base eller smoke")


def to_dict(config) -> dict:
    return asdict(config)
