"""Tests af den regelbaserede PRD-kritiker."""

import random

from tinyprd.critic import critique
from tinyprd.data.synth import DOMAINS, PATTERNS, make_record


def test_critic_likes_a_well_shaped_prd():
    record = make_record(DOMAINS[0], PATTERNS[0], random.Random(1))
    assert critique(record.render()).overall >= 9


def test_critic_flags_common_mistakes():
    bad = "## problem\nwe need a dashboard with slack notifications, it is just a small thing\n## scope\n- dashboard"
    result = critique(bad)
    assert result.scores["problem"] < 10 and result.scores["non-goals"] == 0
    assert any("just" in item for item in result.feedback)
