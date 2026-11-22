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
from ..runtime import count_parameters
from ..sampling import generate_ncm

MAX_IDEA_CHARS = 600

# Forslag på startsiden: (kort etiket, idé). Alle ord findes i ordforrådet.
EXAMPLE_IDEAS = (
    ("Ventende pull requests", "what if we built a tool that lets developers see which pull requests are waiting for "
                               + "reviewers? maybe it could send slack notifications."),
    ("Del dashboards", "idea: something that lets analysts share dashboards with coworkers."),
    ("Påmindelser til lærere", "we should make it easy for teachers to remember when assignments are due."),
    ("Find gamle tickets", "support agents keep asking how to find old tickets. can search be simple?"),
    ("Godkend mange udlæg", "what if finance teams could approve many expense reports at once?"),
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
            "examples": [{"label": label, "idea": idea} for label, idea in EXAMPLE_IDEAS],
            "parameters": {
                "ncm": count_parameters(self.tokenizer) + count_parameters(self.generator),
                "baseline": count_parameters(self.baseline) if self.baseline is not None else None,
            },
        }

    def generate(self, idea: str, temperature: float = 0.0, compare: bool = False) -> dict:
        """Skriv en PRD med NCM, og eventuelt også med baseline til sammenligning."""
        idea = idea.strip()[:MAX_IDEA_CHARS]
        if not idea:
            raise ValueError("Skriv en idé først.")
        prompt = torch.tensor([self.vocab.encode(idea, self.prompt_len)], device=self.device)

        with self._lock, torch.no_grad():
            # Samme seed til begge kørsler, så den tidtagne og den med niveauer giver samme PRD.
            seed = int(torch.randint(0, 2**31 - 1, ()).item())

            # 1) Tidtagning: præcis som modellen kører i praksis.
            torch.manual_seed(seed)
            began = time.perf_counter()
            out = generate_ncm(self.tokenizer, self.generator, prompt, temperature=temperature, top_k=20)
            ids = out.tokens[0].tolist()  # .tolist() venter på GPU'en, så tiden bliver ærlig
            ncm_seconds = time.perf_counter() - began

            # 2) Samme PRD igen, men vi afkoder også teksten efter hvert niveau.
            torch.manual_seed(seed)
            steps = generate_ncm(self.tokenizer, self.generator, prompt, temperature=temperature, top_k=20,
                                 trajectory=True).trajectory
            result = {
                "idea": idea,
                "unknown_words": sorted(set(self.vocab.unknown_words(idea))),
                "ncm": self._describe(idea, ids, out.nfe, ncm_seconds, "NCM"),
                "levels": [
                    {"codes": n, "markdown": to_markdown(self.vocab.decode(step[0].tolist()))}
                    for n, step in zip(self.generator.level_lens, steps)
                ],
            }
            if compare and self.baseline is not None:
                torch.manual_seed(seed)
                began = time.perf_counter()
                tokens, nfe = self.baseline.generate(prompt, self.vocab.eos_id, temperature=temperature, top_k=20)
                ids = tokens[0].tolist()
                seconds = time.perf_counter() - began
                result["baseline"] = self._describe(idea, ids, nfe, seconds, "autoregressive baseline")
        return result

    @staticmethod
    def critique(text: str) -> dict:
        return critique_to_dict(critique(text))

    def _describe(self, idea: str, ids: list[int], nfe: int, seconds: float, model: str) -> dict:
        """Tekst, Markdown, kritik og tal for én models svar."""
        text = self.vocab.decode(ids)
        n_tokens = self._count_tokens(ids)
        result = critique(text)
        label = f"TinyPRD ({model}, {nfe} forward passes) · critic {result.overall:.1f}/10"
        return {
            "text": text,
            "markdown": to_markdown(text, idea, generated_by=label),
            "sections": parse_sections(text),
            "critic": critique_to_dict(result),
            "nfe": nfe,
            "seconds": seconds,
            "tokens": n_tokens,  # ord i PRD'en (frem til <eos>)
            "tokens_per_second": n_tokens / seconds if seconds else 0.0,
            "tokens_per_pass": n_tokens / nfe if nfe else 0.0,
            "ms_per_pass": 1000 * seconds / nfe if nfe else 0.0,
        }

    def _count_tokens(self, ids: list[int]) -> int:
        """Antal rigtige ord: alt før <eos>, uden <pad>."""
        count = 0
        for i in ids:
            if i == self.vocab.eos_id:
                break
            count += i != self.vocab.pad_id
        return count


def critique_to_dict(result: Critique) -> dict:
    return {"scores": result.scores, "overall": result.overall, "feedback": result.feedback}
