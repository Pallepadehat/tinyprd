"""Bindeleddet mellem modellerne og hjemmesiden.

`PrdService` holder de trænede modeller i hukommelsen og svarer med
almindelige Python-dicts, som serveren kan sende som JSON. Den ved intet om
HTTP, så den er nem at teste for sig selv.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from ..critic import Critique, critique
from ..data.prd import parse_sections
from ..data.vocab import Vocab
from ..markdown import to_markdown
from ..models.baseline import AutoregressiveBaseline
from ..models.concept_tokenizer import ConceptTokenizer
from ..models.generator import NextConceptTransformer
from ..sampling import generate_ncm

MAX_IDEA_CHARS = 600

EXAMPLE_IDEAS = (
    (
        "what if we built a tool that lets developers see which pull requests are waiting for reviewers? "
        "maybe it could send slack notifications."
    ),
    "idea: something that lets analysts share dashboards with coworkers.",
    "we should make it easy for teachers to remember when assignments are due.",
    "support agents keep asking how to find old tickets. can search be simple?",
    "what if finance teams could approve many expense reports at once?",
)


@dataclass
class PrdService:
    """De trænede modeller, klar til at blive brugt fra hjemmesiden."""

    vocab: Vocab
    tokenizer: ConceptTokenizer
    generator: NextConceptTransformer
    prompt_len: int
    device: torch.device
    baseline: AutoregressiveBaseline | None = None

    def __post_init__(self) -> None:
        self.tokenizer.eval()
        self.generator.eval()
        if self.baseline is not None:
            self.baseline.eval()
        # Serveren svarer på flere forespørgsler ad gangen. GPU'en skal kun
        # bruges af én ad gangen, så vi låser omkring modelkaldene.
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoints(cls) -> PrdService:
        """Indlæs de modeller, `tinyprd train-all` har gemt i runs/."""
        from ..runtime import load_baseline, load_generator, load_tokenizer, load_vocab, pick_device

        device = pick_device()
        tokenizer, _ = load_tokenizer(device)
        generator, ckpt = load_generator(device)
        try:
            baseline, _ = load_baseline(device)
        except FileNotFoundError:
            baseline = None  # sammenligningen er valgfri
        return cls(load_vocab(), tokenizer, generator, ckpt["data"]["prompt_len"], device, baseline)

    def info(self) -> dict:
        return {
            "device": self.device.type,
            "vocab_size": len(self.vocab),
            "levels": list(self.generator.level_lens),
            "has_baseline": self.baseline is not None,
            "examples": list(EXAMPLE_IDEAS),
        }

    def generate(self, idea: str, temperature: float = 0.0, compare: bool = False) -> dict:
        """Skriv en PRD med NCM, og eventuelt også med baseline til sammenligning."""
        idea = idea.strip()[:MAX_IDEA_CHARS]
        if not idea:
            raise ValueError("Skriv en idé først.")
        prompt = torch.tensor([self.vocab.encode(idea, self.prompt_len)], device=self.device)

        with self._lock, torch.no_grad():
            began = time.perf_counter()
            out = generate_ncm(self.tokenizer, self.generator, prompt, temperature=temperature, top_k=20,
                               trajectory=True)
            ids = out.tokens[0].tolist()  # .tolist() venter på GPU'en, så tiden bliver ærlig
            ncm_seconds = time.perf_counter() - began
            text = self.vocab.decode(ids)
            result = {
                "idea": idea,
                "unknown_words": sorted(set(self.vocab.unknown_words(idea))),
                "ncm": self._describe(idea, text, out.nfe, ncm_seconds, "NCM"),
                "levels": [
                    {"codes": n, "markdown": to_markdown(self.vocab.decode(step[0].tolist()))}
                    for n, step in zip(self.generator.level_lens, out.trajectory)
                ],
            }
            if compare and self.baseline is not None:
                began = time.perf_counter()
                tokens, nfe = self.baseline.generate(prompt, self.vocab.eos_id, temperature=temperature, top_k=20)
                ids = tokens[0].tolist()
                seconds = time.perf_counter() - began
                result["baseline"] = self._describe(idea, self.vocab.decode(ids), nfe, seconds, "autoregressive baseline")
        return result

    @staticmethod
    def critique(text: str) -> dict:
        return critique_to_dict(critique(text))

    @staticmethod
    def _describe(idea: str, text: str, nfe: int, seconds: float, model: str) -> dict:
        result = critique(text)
        label = f"TinyPRD ({model}, {nfe} forward passes) · critic {result.overall:.1f}/10"
        return {
            "text": text,
            "markdown": to_markdown(text, idea, generated_by=label),
            "sections": parse_sections(text),
            "critic": critique_to_dict(result),
            "nfe": nfe,
            "seconds": seconds,
        }


def critique_to_dict(result: Critique) -> dict:
    return {"scores": result.scores, "overall": result.overall, "feedback": result.feedback}
