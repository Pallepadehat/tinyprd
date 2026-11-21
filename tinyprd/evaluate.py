"""Evaluering: NCM mod den autoregressive baseline.

For hvert testsæt genererer begge modeller en PRD for hver idé, og vi måler:

    critic      kritikerens samlede score (0-10), se critic.py
    word_f1     ord-overlap med facit-PRD'en (0-1)
    intent      andel hvor intent-linjen er præcis rigtig
    nfe         antal forward passes pr. PRD (lavere = hurtigere)
    sek/prd     målt tid pr. PRD på din maskine

To testsæt:
    test_seen    nye eksempler af kombinationer modellen har set
    test_unseen  kombinationer (domæne x mønster) den ALDRIG har set
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import torch

from .config import RUNS_DIR, DataConfig
from .critic import critique
from .prd import parse_sections
from .runtime import (
    encode_records,
    load_baseline,
    load_generator,
    load_split,
    load_tokenizer,
    load_vocab,
    pick_device,
    seed_everything,
)
from .sampling import generate_ncm
from .training import tokenizer_accuracy


def word_f1(predicted: str, reference: str) -> float:
    """F1 over ord-tællinger: 1.0 betyder samme ord i samme mængder."""
    pred, ref = Counter(predicted.split()), Counter(reference.split())
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def intent_of(text: str) -> str:
    return " ".join(parse_sections(text).get("intent", [""])[:1])


def score_texts(predictions: list[str], references: list[str]) -> dict[str, float]:
    n = len(predictions)
    return {
        "critic": sum(critique(p).overall for p in predictions) / n,
        "word_f1": sum(word_f1(p, r) for p, r in zip(predictions, references)) / n,
        "intent": sum(intent_of(p) == intent_of(r) for p, r in zip(predictions, references)) / n,
    }


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(n_per_split: int = 60, batch_size: int = 32, runs_dir: Path = RUNS_DIR) -> dict:
    seed_everything(0)
    device = pick_device()
    vocab = load_vocab()
    tokenizer, _ = load_tokenizer(device, runs_dir)
    generator, gen_ckpt = load_generator(device, runs_dir)
    data_cfg = DataConfig(**gen_ckpt["data"])
    try:
        baseline, _ = load_baseline(device, runs_dir)
    except FileNotFoundError:
        baseline = None
        print("(Ingen baseline trænet. Springer sammenligningen over.)")

    results: dict = {}
    for split in ("test_seen", "test_unseen"):
        records = load_split(split)[:n_per_split]
        prompts, targets = encode_records(records, vocab, data_cfg)
        prompts = prompts.to(device)
        references = [vocab.decode(t.tolist()) for t in targets]

        split_results = {
            "tokenizer_acc_per_level": tokenizer_accuracy(tokenizer, targets.to(device), vocab.pad_id),
        }

        models = {"ncm": "ncm"} | ({"baseline": "baseline"} if baseline else {})
        for name in models:
            texts, nfe_total, seconds = [], 0, 0.0
            for start in range(0, len(prompts), batch_size):
                batch = prompts[start : start + batch_size]
                _sync(device)
                began = time.perf_counter()
                if name == "ncm":
                    out = generate_ncm(tokenizer, generator, batch, vocab.pad_id)
                    tokens, nfe = out.tokens, out.nfe
                else:
                    tokens, nfe = baseline.generate(batch, vocab.eos_id)
                _sync(device)
                seconds += time.perf_counter() - began
                nfe_total += nfe * len(batch)
                texts += [vocab.decode(t.tolist()) for t in tokens]

            split_results[name] = score_texts(texts, references) | {
                "nfe": nfe_total / len(texts),
                "sec_per_prd": seconds / len(texts),
                "example": texts[0],
            }
        results[split] = split_results
        print_split(split, split_results)

    (runs_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nGemt: {runs_dir / 'results.json'}")
    return results


def print_split(split: str, results: dict) -> None:
    print(f"\n=== {split} ===")
    accuracy = results["tokenizer_acc_per_level"]
    print("Tokenizer: ord gendannet efter niveau 1..K: " + "  ".join(f"{a:.0%}" for a in accuracy))
    print(f"{'model':<10} {'critic':>7} {'word_f1':>8} {'intent':>7} {'nfe':>6} {'sek/prd':>9}")
    for name in ("ncm", "baseline"):
        if name in results:
            r = results[name]
            print(f"{name:<10} {r['critic']:7.2f} {r['word_f1']:8.3f} {r['intent']:7.0%} {r['nfe']:6.1f} {r['sec_per_prd']:9.4f}")
