"""Tests af Markdown-eksporten (prd.md)."""

import random

from tinyprd.critic import critique
from tinyprd.data.synth import DOMAINS, PATTERNS, make_record
from tinyprd.markdown import to_markdown


def test_markdown_is_a_clean_prd_document():
    record = make_record(DOMAINS[0], PATTERNS[0], random.Random(0))
    markdown = to_markdown(record.render(), record.idea, generated_by="TinyPRD")
    assert markdown.startswith("# PRD: Pull request wait visibility\n")
    assert f"> **Idea:** {record.idea}" in markdown
    assert "| **Appetite** |" in markdown
    for heading in ("## Problem", "## Users", "## Success metrics", "## Non-goals", "## Open questions"):
        assert heading in markdown
    assert "- **Primary:** developers" in markdown
    # Den færdige prd.md skal stadig være en god PRD ifølge kritikeren.
    assert critique(markdown).overall >= 9


def test_markdown_survives_messy_output():
    markdown = to_markdown("noise ## problem\nfoo bar ## weird thing\n- x\n- scope\n- a thing")
    assert markdown.startswith("# PRD: Untitled")
    assert "## Problem\n\nFoo bar." in markdown
    assert "## Weird thing\n\nX." in markdown
    assert "## Scope\n\n- A thing" in markdown  # "- scope" uden ## er en tabt overskrift
