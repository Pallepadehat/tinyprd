"""Kommandolinjen.

    tinyprd data                  lav det syntetiske datasæt
    tinyprd train-all             træn tokenizer, generator og baseline
    tinyprd generate "idé"        skriv en PRD
    tinyprd generate "idé" --levels   vis hvordan PRD'en bliver skarpere niveau for niveau
    tinyprd critique fil.md       giv din egen PRD point
    tinyprd evaluate              sammenlign NCM med baseline
    tinyprd serve                 åbn webinterfacet på http://localhost:8000

    prd "idé"                     form idéen og gem den færdige PRD som prd.md
    prd "idé" --out min_prd.md    gem et andet sted
    prd "idé" --full              vis hele PRD'en som Markdown i terminalen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .config import DATA_DIR, PRESET_NAMES, get_preset
from .critic import critique
from .data.prd import parse_sections
from .markdown import to_markdown


def cmd_data(args) -> None:
    from .data.synth import build_dataset, load_records
    from .data.vocab import Vocab

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
    out = generate_ncm(tokenizer, generator, prompt, temperature=temperature, top_k=20, trajectory=show_levels)
    if show_levels:
        for k, tokens in enumerate(out.trajectory):
            print(f"──────── efter niveau {k + 1}/{len(out.trajectory)} "
                  f"({generator.level_lens[k]} koder) ────────")
            print(vocab.decode(tokens[0].tolist()))
            print()
    return vocab.decode(out.tokens[0].tolist()), out.nfe


def _markdown(idea: str, text: str, nfe: int) -> str:
    score = critique(text).overall
    return to_markdown(text, idea, generated_by=f"TinyPRD (NCM, {nfe} forward passes) · critic {score:.1f}/10")


def _save(markdown: str, wanted: str) -> Path:
    """Gem uden at overskrive: findes prd.md, bliver det prd-2.md, prd-3.md …"""
    path, n = Path(wanted), 2
    while path.exists():
        path = Path(wanted).with_stem(f"{Path(wanted).stem}-{n}")
        n += 1
    path.write_text(markdown)
    return path


def cmd_generate(args) -> None:
    text, nfe = _generate(args.idea, args.temperature, args.levels)
    if not args.levels:
        print(text)
    print(f"\n──────── kritiker ({nfe} forward passes) ────────")
    print(critique(text).report())
    if args.out:
        print(f"\nGemt som Markdown: {_save(_markdown(args.idea, text, nfe), args.out)}")


def cmd_critique(args) -> None:
    text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text()
    print(critique(text).report())


def cmd_evaluate(args) -> None:
    from .evaluate import evaluate

    evaluate(n_per_split=args.n)


def cmd_serve(args) -> None:
    from .web.server import serve

    serve(args.host, args.port, open_browser=args.open)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinyprd", description="TinyPRD: en lille next-concept model der former produktidéer til PRD'er.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="lav det syntetiske datasæt")
    p.add_argument("--preset", default="large", choices=PRESET_NAMES)
    p.set_defaults(func=cmd_data)

    for stage in ("tokenizer", "generator", "baseline", "all"):
        p = sub.add_parser(f"train-{stage}", help=f"træn {stage}")
        p.add_argument("--preset", default="large", choices=PRESET_NAMES)
        p.set_defaults(func=cmd_train, stage=stage)

    p = sub.add_parser("generate", help="skriv en PRD ud fra en idé")
    p.add_argument("idea")
    p.add_argument("--levels", action="store_true", help="vis teksten efter hvert niveau")
    p.add_argument("--temperature", type=float, default=0.0, help="0 = deterministisk, fx 0.8 = mere variation")
    p.add_argument("--out", help="gem PRD'en som Markdown, fx prd.md")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("critique", help="giv en PRD point (fil eller - for stdin)")
    p.add_argument("file")
    p.set_defaults(func=cmd_critique)

    p = sub.add_parser("evaluate", help="sammenlign NCM med baseline på testsættene")
    p.add_argument("--n", type=int, default=60, help="antal eksempler pr. testsæt")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("serve", help="start webinterfacet i browseren")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = kun din egen maskine")
    p.add_argument("--open", action="store_true", help="åbn browseren automatisk")
    p.set_defaults(func=cmd_serve)
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
    parser.add_argument("--full", action="store_true", help="vis hele PRD'en som Markdown")
    parser.add_argument("--out", default="prd.md", help="hvor PRD'en gemmes (standard: prd.md)")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args(argv)

    text, nfe = _generate(args.idea, args.temperature, show_levels=False)
    markdown = _markdown(args.idea, text, nfe)
    saved = _save(markdown, args.out)
    if args.full:
        print(markdown)
        print(f"Gemt: {saved}")
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
    print(f"Hele PRD'en er gemt som Markdown: {saved}")


if __name__ == "__main__":
    main()
