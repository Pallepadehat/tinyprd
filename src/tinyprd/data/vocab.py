"""En lille ord-baseret tekst-tokenizer.

TinyPRD arbejder i et snævert domæne, så et simpelt ordforråd (et par hundrede
ord) er nok og meget lettere at forstå end BPE: én token er ét ord, ét
tegn eller ét linjeskift.

Om navngivning: NCM-artiklen kalder også sin trin-1-model for en "tokenizer".
For at undgå forvirring hedder denne fil `Vocab` (tekst <-> ord-id'er), og
trin 1 hedder `ConceptTokenizer` (ord-id'er <-> hierarki af koder).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, EOS, NEWLINE = "<pad>", "<unk>", "<eos>", "<nl>"
SPECIAL_TOKENS = (PAD, UNK, EOS, NEWLINE)

# "##"-overskrifter, linjeskift, ord (med bindestreg som "non-goals"),
# tal, og ellers ét tegnsætningstegn ad gangen.
_TOKEN_PATTERN = re.compile(r"##|\n|[a-z0-9]+(?:-[a-z0-9]+)*|[^\sa-z0-9]")

# Tegn der sættes direkte efter forrige ord, når tokens bliver til tekst.
_NO_SPACE_BEFORE = {".", ",", "?", "!", ":", ";", ")"}


def split_words(text: str) -> list[str]:
    """'Hello, world!' -> ['hello', ',', 'world', '!'] (med små bogstaver)."""
    return [NEWLINE if t == "\n" else t for t in _TOKEN_PATTERN.findall(text.lower())]


class Vocab:
    """Oversætter ord til heltal (id'er) og tilbage."""

    def __init__(self, words: list[str]):
        self.words = list(words)
        self.index = {word: i for i, word in enumerate(self.words)}
        self.pad_id = self.index[PAD]
        self.unk_id = self.index[UNK]
        self.eos_id = self.index[EOS]

    def __len__(self) -> int:
        return len(self.words)

    # ---- bygge og gemme -------------------------------------------

    @classmethod
    def build(cls, texts: list[str], min_count: int = 1) -> Vocab:
        counts = Counter(word for text in texts for word in split_words(text))
        words = sorted(w for w, c in counts.items() if c >= min_count and w not in SPECIAL_TOKENS)
        return cls(list(SPECIAL_TOKENS) + words)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.words, indent=0))

    @classmethod
    def load(cls, path: Path) -> Vocab:
        return cls(json.loads(path.read_text()))

    # ---- kode og afkode -----------------------------------------

    def encode(self, text: str, length: int | None = None, add_eos: bool = False) -> list[int]:
        """Tekst -> id'er. Med `length` klippes der, eller der fyldes op med <pad>."""
        ids = [self.index.get(word, self.unk_id) for word in split_words(text)]
        if add_eos:
            ids.append(self.eos_id)
        if length is not None:
            ids = ids[:length] + [self.pad_id] * max(0, length - len(ids))
        return ids

    def unknown_words(self, text: str) -> list[str]:
        """Ord i `text`, som modellen aldrig har set (de bliver til <unk>)."""
        return [w for w in split_words(text) if w not in self.index]

    def decode(self, ids: list[int]) -> str:
        """Id'er -> læsbar tekst. Stopper ved <eos> og springer <pad> over."""
        pieces: list[str] = []
        for i in ids:
            word = self.words[i]
            if word == EOS:
                break
            if word == PAD:
                continue
            if word == NEWLINE:
                pieces.append("\n")
            elif not pieces or pieces[-1] == "\n" or word in _NO_SPACE_BEFORE:
                pieces.append(word)
            else:
                pieces.append(" " + word)
        return "".join(pieces)
