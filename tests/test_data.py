"""Tests af data-delen: PRD-formatet, datasættet og ordforrådet."""

import random

from tinyprd.data.prd import SECTION_ORDER, parse_sections
from tinyprd.data.synth import DOMAINS, PATTERNS, build_dataset, load_records, make_record
from tinyprd.data.vocab import Vocab


def test_record_renders_every_section_in_order():
    record = make_record(DOMAINS[0], PATTERNS[0], random.Random(0))
    sections = parse_sections(record.render())
    assert list(sections) == list(SECTION_ORDER)
    assert all(sections[name] for name in SECTION_ORDER)


def test_unseen_combinations_never_appear_in_training(tmp_path):
    build_dataset(tmp_path, examples_per_combo=1)
    train = {(r.domain, r.pattern) for r in load_records(tmp_path / "train.jsonl")}
    unseen = {(r.domain, r.pattern) for r in load_records(tmp_path / "test_unseen.jsonl")}
    assert unseen and not (train & unseen)


def test_vocab_round_trip():
    text = "## problem\ndevelopers wait for reviewers, every day?"
    vocab = Vocab.build([text])
    assert vocab.decode(vocab.encode(text)) == text
    assert len(vocab.encode(text, length=40)) == 40


def test_parser_tolerates_messy_model_output():
    sections = parse_sections("## problem\nfoo bar. ## users\nprimary: x.\n## appetite one week.")
    assert sections == {"problem": ["foo bar."], "users": ["primary: x."], "appetite": ["one week."]}
