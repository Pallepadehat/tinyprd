"""Hurtige tests, der tjekker at byggeklodserne opfører sig som forventet.

Kør dem med:  pytest
De bruger små modeller på CPU, så de tager kun få sekunder.
"""

import random

import torch

from tinyprd.concept_tokenizer import ConceptTokenizer
from tinyprd.config import ConceptTokenizerConfig, GeneratorConfig
from tinyprd.critic import critique
from tinyprd.generator import NextConceptTransformer, corrupt_codes
from tinyprd.layers import downsample_matrix, finest_positions, upsample_matrix
from tinyprd.prd import SECTION_ORDER, parse_sections
from tinyprd.sampling import generate_ncm
from tinyprd.synth import DOMAINS, PATTERNS, build_dataset, load_records, make_record
from tinyprd.vocab import Vocab

LEVELS = (2, 4, 8, 16)  # et lille hierarki til tests


def small_models(vocab_size=50, prompt_len=6):
    tok_cfg = ConceptTokenizerConfig(level_lens=LEVELS, codebook_size=32, d_latent=16, n_heads=2)
    tokenizer = ConceptTokenizer(tok_cfg, vocab_size, target_len=LEVELS[-1])
    gen_cfg = GeneratorConfig(d_model=32, n_layers=1, n_heads=2)
    generator = NextConceptTransformer(gen_cfg, vocab_size, prompt_len, LEVELS[-1], LEVELS, 16, 32)
    return tokenizer, generator


# ---- data -----------------------------------------------------------------


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


# ---- lag ------------------------------------------------------------------


def test_resize_matrices_are_averages():
    down, up = downsample_matrix(256, 8), upsample_matrix(8, 256)
    assert torch.allclose(down.sum(1), torch.ones(8))  # hver plads er et gennemsnit
    assert torch.allclose(up.sum(1), torch.ones(256))
    assert finest_positions(4, 256).tolist() == [32, 96, 160, 224]


# ---- koncept-tokenizer ------------------------------------------------------


def test_tokenizer_produces_one_code_list_per_level():
    tokenizer, _ = small_models()
    tokenizer.eval()  # slå dropout fra, så resultaterne kan sammenlignes
    tokens = torch.randint(0, 50, (3, LEVELS[-1]))
    codes = tokenizer.tokenize(tokens)
    assert [c.shape for c in codes] == [torch.Size([3, n]) for n in LEVELS]
    # At genopbygge fra koderne giver samme resultat som quantize.
    _, partials, _ = tokenizer.quantize(tokenizer.encode(tokens))
    assert torch.allclose(tokenizer.partials_from_codes(codes)[-1], partials[-1], atol=1e-5)


def test_tokenizer_training_step_has_gradients():
    tokenizer, _ = small_models()
    losses = tokenizer(torch.randint(0, 50, (4, LEVELS[-1])), pad_id=0)
    losses["loss"].backward()
    assert tokenizer.word_embed.weight.grad is not None


# ---- generator --------------------------------------------------------------


def test_mask_is_block_diagonal():
    _, generator = small_models()
    segments = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # 2 idé-ord, niveau 1 (2), niveau 2 (3)
    mask = generator.attention_mask(segments, torch.tensor([[True, False]]))[0, 0]
    assert mask[2, 3] and not mask[2, 4]  # niveau 1 ser sig selv, ikke niveau 2
    assert mask[5, 0] and not mask[5, 1]  # alle ser rigtige idé-ord, ikke <pad>


def test_generator_trains_and_generates():
    tokenizer, generator = small_models()
    prompt = torch.randint(1, 50, (2, 6))
    codes = tokenizer.tokenize(torch.randint(0, 50, (2, LEVELS[-1])))
    partials = tokenizer.partials_from_codes(corrupt_codes(codes, 32, 0.5, 0.1))
    loss = generator(prompt, 0, partials, codes)
    assert loss.item() > 0

    out = generate_ncm(tokenizer.eval(), generator.eval(), prompt, pad_id=0, trajectory=True)
    assert out.tokens.shape == (2, LEVELS[-1])
    assert out.nfe == len(LEVELS) == len(out.trajectory)


# ---- kritiker ---------------------------------------------------------------


def test_critic_likes_a_well_shaped_prd():
    record = make_record(DOMAINS[0], PATTERNS[0], random.Random(1))
    assert critique(record.render()).overall >= 9


def test_critic_flags_common_mistakes():
    bad = "## problem\nwe need a dashboard with slack notifications, it is just a small thing\n## scope\n- dashboard"
    result = critique(bad)
    assert result.scores["problem"] < 10 and result.scores["non-goals"] == 0
    assert any("just" in item for item in result.feedback)
