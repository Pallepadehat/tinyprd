"""Fælles hjælpere til testene.

Testene bruger små, utrænede modeller på CPU, så de tager kun få sekunder.
"""

import pytest

from tinyprd.config import ConceptTokenizerConfig, GeneratorConfig
from tinyprd.models.concept_tokenizer import ConceptTokenizer
from tinyprd.models.generator import NextConceptTransformer

SMALL_LEVELS = (2, 4, 8, 16)  # et lille hierarki til tests
SMALL_VOCAB = 50
SMALL_PROMPT = 6


@pytest.fixture
def small_models():
    """(tokenizer, generator) i mini-størrelse."""
    tok_cfg = ConceptTokenizerConfig(level_lens=SMALL_LEVELS, codebook_size=32, d_latent=16, n_heads=2)
    tokenizer = ConceptTokenizer(tok_cfg, SMALL_VOCAB, target_len=SMALL_LEVELS[-1])
    gen_cfg = GeneratorConfig(d_model=32, n_layers=1, n_heads=2)
    generator = NextConceptTransformer(gen_cfg, SMALL_VOCAB, SMALL_PROMPT, SMALL_LEVELS[-1], SMALL_LEVELS, 16, 32)
    return tokenizer, generator
