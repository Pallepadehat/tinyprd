"""Kommandolinjen.

    tinyprd data                  lav det syntetiske datasæt
    tinyprd train-all             træn tokenizer, generator og baseline
    tinyprd generate "idé"        skriv en PRD
    tinyprd generate "idé" --levels   vis hvordan PRD'en bliver skarpere niveau for niveau
    tinyprd critique fil.md       giv din egen PRD point
    tinyprd evaluate              sammenlign NCM med baseline

    prd "idé"                     kort version: forme idéen (problem, mål, scope, spørgsmål)
    prd "idé" --full              hele PRD'en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import DATA_DIR, get_preset
from .critic import critique
from .prd import parse_sections


def cmd_data(args) -> None:
    from .synth import build_dataset, load_records
    from .vocab import Vocab

    preset = get_preset(args.preset)
    sizes = build_dataset(DATA_DIR, examples_per_combo=preset.data.examples_per_combo, seed=preset.data.seed)
    # Ordforrådet bygges kun af træningsdata, så testsættene er ærlige.
    train = load_records(DATA_DIR / "train.jsonl")
    vocab = Vocab.build([r.render() for r in train] + [r.idea for r in train])
    vocab.save(DATA_DIR / "vocab.json")
    print(f"Datasæt gemt i {DATA_DIR}: {sizes}, ordforråd: {len(vocab)} ord")


def cmd_train(args) -> None:
    from . import training

    preset = get_preset(args.preset)
    if args.stage in ("tokenizer", "all"):
        training.train_tokenizer(preset)
    if args.stage in ("generator", "all"):
        training.train_generator(preset)
    if args.stage in ("baseline", "all"):
        training.train_baseline(preset)


def _load_ncm():
    from .runtime import load_generator, load_tokenizer, load_vocab, pick_device

    device = pick_device()
    vocab = load_vocab()
    tokenizer, _ = load_tokenizer(device)
    generator, ckpt = load_generator(device)
    return device, vocab, tokenizer, generator, ckpt["data"]["prompt_len"]


def _generate(idea: str, temperature: float, show_levels: bool):
    from .sampling import generate_ncm

    device, vocab, tokenizer, generator, prompt_len = _load_ncm()
    unknown = vocab.unknown_words(idea)
    if unknown:
        print(f"(Ukendte ord bliver til <unk>: {', '.join(sorted(set(unknown)))})\n", file=sys.stderr)
    prompt = torch.tensor([vocab.encode(idea, prompt_len)], device=device)
    out = generate_ncm(tokenizer, generator, prompt, vocab.pad_id, temperature=temperature, top_k=20, trajectory=show_levels)
    if show_levels:
        for k, tokens in enumerate(out.trajectory):
            print(f"──────── efter niveau {k + 1}/{len(out.trajectory)} "
                  f"({generator.level_lens[k]} koder) ────────")
            print(vocab.decode(tokens[0].tolist()))
            print()
    return vocab.decode(out.tokens[0].tolist()), out.nfe


def cmd_generate(args) -> None:
    text, nfe = _generate(args.idea, args.temperature, args.levels)
    if not args.levels:
        print(text)
    print(f"\n──────── kritiker ({nfe} forward passes) ────────")
    print(critique(text).report())


def cmd_critique(args) -> None:
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    print(critique(text).report())


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate

    evaluate(n_per_split=args.n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyprd", description="TinyPRD: en lille next-concept model der former produktidéer til PRD'er.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="lav det syntetiske datasæt")
    p.add_argument("--preset", default="tiny", choices=["tiny", "base", "smoke"])
    p.set_defaults(func=cmd_data)

    for stage in ("tokenizer", "generator", "baseline", "all"):
        p = sub.add_parser(f"train-{stage}", help=f"træn {stage}")
        p.add_argument("--preset", default="tiny", choices=["tiny", "base", "smoke"])
        p.set_defaults(func=cmd_train, stage=stage)

    p = sub.add_parser("generate", help="skriv en PRD ud fra en idé")
    p.add_argument("idea")
    p.add_argument("--levels", action="store_true", help="vis teksten efter hvert niveau")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = deterministisk, fx 0.8 = mere variation")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("critique", help="giv en PRD point (fil eller - for stdin)")
    p.add_argument("file")
    p.set_defaults(func=cmd_critique)

    p = sub.add_parser("evaluate", help="sammenlign NCM med baseline på testsættene")
    p.add_argument("--n", type=int, default=60, help="antal eksempler pr. testsæt")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


# ---- `prd`: den korte "forme idéen" visning --------------------------------

SHAPE_VIEW = (
    ("Problem", "problem"),
    ("Mulige succesmål", "success metrics"),
    ("Med i scope", "scope"),
    ("Ikke med (non-goals)", "non-goals"),
    ("Tidsbudget (appetite)", "appetite"),
    ("Åbne spørgsmål", "open questions"),
)


def prd_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="prd", description="Form en produktidé, før du skriver PRD'en.")
    parser.add_argument("idea")
    parser.add_argument("--full", action="store_true", help="vis hele PRD'en")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    text, _ = _generate(args.idea, args.temperature, show_levels=False)
    if args.full:
        print(text)
        return

    sections = parse_sections(text)
    intent = " ".join(sections.get("intent", []))
    print(f"Lad os forme idéen, før vi skriver PRD'en: {intent}\n")
    for title, key in SHAPE_VIEW:
        lines = sections.get(key, [])
        if not lines:
            continue
        print(title)
        print("─" * len(title))
        for line in lines:
            print(f"• {line}" if len(lines) > 1 or key not in ("problem", "appetite") else line)
        print()
    print("Kør med --full for at få hele PRD'en.")


if __name__ == "__main__":
    main()
