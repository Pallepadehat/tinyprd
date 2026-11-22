"""Tests af modellerne: pres/stræk, koncept-tokenizer, generator og generering."""

import torch
from conftest import SMALL_LEVELS

from tinyprd.models.generator import corrupt_codes
from tinyprd.models.layers import downsample_matrix, finest_positions, upsample_matrix
from tinyprd.sampling import collapse_repeats, generate_ncm

# ---- lag ------------------------------------------------------------------


def test_resize_matrices_are_averages():
    down, up = downsample_matrix(256, 8), upsample_matrix(8, 256)
    assert torch.allclose(down.sum(1), torch.ones(8))  # hver plads er et gennemsnit
    assert torch.allclose(up.sum(1), torch.ones(256))
    assert finest_positions(4, 256).tolist() == [32, 96, 160, 224]


# ---- koncept-tokenizer ------------------------------------------------------


def test_tokenizer_produces_one_code_list_per_level(small_models):
    tokenizer, _ = small_models
    tokenizer.eval()  # slå dropout fra, så resultaterne kan sammenlignes
    tokens = torch.randint(0, 50, (3, SMALL_LEVELS[-1]))
    codes = tokenizer.tokenize(tokens)
    assert [c.shape for c in codes] == [torch.Size([3, n]) for n in SMALL_LEVELS]
    # At genopbygge fra koderne giver samme resultat som quantize.
    _, partials, _ = tokenizer.quantize(tokenizer.encode(tokens))
    assert torch.allclose(tokenizer.partials_from_codes(codes)[-1], partials[-1], atol=1e-5)


def test_tokenizer_training_step_has_gradients(small_models):
    tokenizer, _ = small_models
    losses = tokenizer(torch.randint(0, 50, (4, SMALL_LEVELS[-1])), pad_id=0)
    losses["loss"].backward()
    assert tokenizer.word_embed.weight.grad is not None


# ---- generator --------------------------------------------------------------


def test_mask_is_block_diagonal(small_models):
    _, generator = small_models
    segments = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # 2 idé-ord, niveau 1 (2), niveau 2 (3)
    mask = generator.attention_mask(segments)
    assert mask[2, 3] and not mask[2, 4]  # niveau 1 ser sig selv, ikke niveau 2
    assert mask[5, 0] and mask[5, 1]  # alle må se idéen
    assert not mask[0, 2]  # idéen ser ikke niveauerne


def test_generator_trains_and_generates(small_models):
    tokenizer, generator = small_models
    prompt = torch.randint(1, 50, (2, 6))
    codes = tokenizer.tokenize(torch.randint(0, 50, (2, SMALL_LEVELS[-1])))
    partials = tokenizer.partials_from_codes(corrupt_codes(codes, 32, 0.5, 0.1))
    loss = generator(prompt, partials, codes)
    assert loss.item() > 0

    out = generate_ncm(tokenizer.eval(), generator.eval(), prompt, trajectory=True)
    assert out.tokens.shape == (2, SMALL_LEVELS[-1])
    assert out.nfe == len(SMALL_LEVELS) == len(out.trajectory)


def test_collapse_repeats_removes_stutter():
    cleaned = collapse_repeats(torch.tensor([[5, 5, 6, 7, 7, 7, 2]]))
    assert cleaned.tolist() == [[5, 6, 7, 2, 2, 2, 2]]  # formen bevares


def test_level_by_level_backward_gives_same_gradients(small_models):
    tokenizer, generator = small_models
    generator.eval()  # uden dropout, så de to kørsler kan sammenlignes
    prompt = torch.randint(1, 50, (2, 6))
    codes = tokenizer.eval().tokenize(torch.randint(0, 50, (2, SMALL_LEVELS[-1])))
    with torch.no_grad():  # som i træningen: tokenizeren er frosset
        partials = tokenizer.partials_from_codes(codes)

    generator(prompt, partials, codes).backward()
    expected = [p.grad.clone() for p in generator.parameters()]
    generator.zero_grad()
    generator.backward_level_by_level(prompt, partials, codes)
    for grad, p in zip(expected, generator.parameters()):
        assert torch.allclose(grad, p.grad, atol=1e-6)
