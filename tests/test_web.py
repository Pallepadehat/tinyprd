"""Tests af webinterfacet: service-laget og HTTP-serveren (med utrænede mini-modeller)."""

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch
from conftest import SMALL_LEVELS, SMALL_PROMPT, SMALL_VOCAB

from tinyprd.data.vocab import SPECIAL_TOKENS, Vocab
from tinyprd.web.server import make_server
from tinyprd.web.service import PrdService


@pytest.fixture
def service(small_models):
    tokenizer, generator = small_models
    # Præcis SMALL_VOCAB ord, så alle id'er modellen kan vælge, kan afkodes.
    words = list(SPECIAL_TOKENS) + ["##", "problem", "developers", "wait"]
    words += [f"word{i}" for i in range(SMALL_VOCAB - len(words))]
    return PrdService(Vocab(words), tokenizer, generator, SMALL_PROMPT, torch.device("cpu"))


def test_service_returns_prd_levels_and_critique(service):
    result = service.generate("developers wait for reviewers")
    assert len(result["levels"]) == len(SMALL_LEVELS)
    assert result["ncm"]["nfe"] == len(SMALL_LEVELS)
    assert 0 <= result["ncm"]["critic"]["overall"] <= 10
    assert result["unknown_words"] == ["for", "reviewers"]
    stats = result["ncm"]
    assert stats["tokens"] <= SMALL_LEVELS[-1] and stats["tokens_per_second"] >= 0
    assert stats["markdown"].startswith("# PRD:")


def test_service_rejects_empty_idea(service):
    with pytest.raises(ValueError):
        service.generate("   ")


def test_http_server_end_to_end(service):
    server = make_server(service, port=0)  # port 0 = vælg en ledig port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, body):
        request = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(request).read())

    try:
        assert b"TinyPRD" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/info").read())["levels"] == list(SMALL_LEVELS)
        assert post("/api/generate", {"idea": "developers wait"})["ncm"]["markdown"].startswith("# PRD:")
        assert post("/api/critique", {"text": "## problem\nfoo"})["scores"]["non-goals"] == 0
        with pytest.raises(urllib.error.HTTPError) as error:
            post("/api/generate", {"idea": ""})
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
