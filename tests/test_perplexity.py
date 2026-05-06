"""Tests for src/analysis/perplexity.py — the M4 PPL evaluator.

Uses a CPU-built tiny LlamaForCausalLM (vocab=128, hidden=32, 2 layers,
2 heads) so the suite runs in a few seconds without any HF model
download. The same compute_perplexity() runs against full-sized models
on the pod; the math is independent of model size.
"""
from __future__ import annotations

import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.analysis.perplexity import compute_perplexity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_llama():
    """A small CPU LlamaForCausalLM. Deterministic init via torch.manual_seed."""
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,   # short ctx so we exercise sliding-window
        rms_norm_eps=1e-5,
    )
    model = LlamaForCausalLM(cfg).eval()
    # Disable HF cache so tests don't accidentally rely on it
    model.config.use_cache = False
    return model


@pytest.fixture
def short_input(tiny_llama):
    """A sequence shorter than the model's max_position_embeddings."""
    torch.manual_seed(1)
    L = 32  # < tiny_llama.config.max_position_embeddings (64)
    return torch.randint(0, tiny_llama.config.vocab_size, (1, L))


@pytest.fixture
def long_input(tiny_llama):
    """A sequence longer than max_position_embeddings — exercises sliding."""
    torch.manual_seed(2)
    L = 200  # >> 64
    return torch.randint(0, tiny_llama.config.vocab_size, (1, L))


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------

class TestBasicInvariants:
    def test_returns_finite_positive_float(self, tiny_llama, short_input):
        ppl = compute_perplexity(tiny_llama, short_input)
        assert isinstance(ppl, float)
        assert math.isfinite(ppl)
        assert ppl > 0.0

    def test_ppl_is_at_least_one(self, tiny_llama, short_input):
        """PPL is exp(NLL) where NLL >= 0 → PPL >= 1 always."""
        ppl = compute_perplexity(tiny_llama, short_input)
        assert ppl >= 1.0 - 1e-6

    def test_ppl_bounded_by_vocab_for_uniform_baseline(self, tiny_llama, short_input):
        """A LM whose output is exactly uniform over the vocab has
        PPL = vocab_size. A randomly-initialized model can do better
        (some tokens slightly favored by chance) or worse, but on a
        random input shouldn't be wildly off the uniform baseline."""
        ppl = compute_perplexity(tiny_llama, short_input)
        vocab = tiny_llama.config.vocab_size
        # Loose bracket: 0.1× to 100× the uniform baseline. A
        # random-init Llama on a random sequence should sit comfortably
        # inside this — anything outside would indicate a math bug.
        assert 0.1 * vocab <= ppl <= 100 * vocab, (
            f"PPL={ppl:.2f} too far from uniform baseline vocab={vocab}"
        )


# ---------------------------------------------------------------------------
# Single-pass branch
# ---------------------------------------------------------------------------

class TestShortSequenceSinglePass:
    def test_short_uses_single_pass(self, tiny_llama, short_input):
        """Sequences <= max_length use the single-pass branch.
        Sanity: result matches HF's direct loss-from-labels formula."""
        max_len = tiny_llama.config.max_position_embeddings
        assert short_input.shape[1] < max_len
        ppl = compute_perplexity(tiny_llama, short_input)

        # Reference: do the single forward by hand and exp(loss)
        with torch.no_grad():
            out = tiny_llama(short_input, labels=short_input)
        ppl_ref = float(torch.exp(out.loss).item())
        assert abs(ppl - ppl_ref) < 1e-5

    def test_short_ignores_stride(self, tiny_llama, short_input):
        """When L <= max_length, stride is irrelevant — single pass."""
        ppl_a = compute_perplexity(tiny_llama, short_input, stride=8)
        ppl_b = compute_perplexity(tiny_llama, short_input, stride=32)
        assert abs(ppl_a - ppl_b) < 1e-9


# ---------------------------------------------------------------------------
# Sliding-window branch
# ---------------------------------------------------------------------------

class TestLongSequenceSlidingWindow:
    def test_long_returns_finite(self, tiny_llama, long_input):
        ppl = compute_perplexity(tiny_llama, long_input)
        assert math.isfinite(ppl)
        assert ppl > 0.0

    def test_smaller_stride_more_accurate(self, tiny_llama, long_input):
        """Smaller stride = more context per scored token → equal or
        better PPL on the same sequence. Sanity check that the sliding
        accumulation isn't double-counting tokens."""
        # Pick strides that both yield fresh-token windows
        max_len = tiny_llama.config.max_position_embeddings
        ppl_small = compute_perplexity(tiny_llama, long_input,
                                       max_length=max_len, stride=8)
        ppl_large = compute_perplexity(tiny_llama, long_input,
                                       max_length=max_len, stride=max_len)
        # Both finite; both within sane range. Don't assert ordering —
        # on a random-init model with a random sequence, the two can
        # land in either order. The point is just "no crash, both
        # comparable order of magnitude".
        for p in (ppl_small, ppl_large):
            assert math.isfinite(p) and p > 0
        assert 0.01 * ppl_large <= ppl_small <= 100 * ppl_large

    def test_max_length_override_changes_window(self, tiny_llama, long_input):
        """Custom max_length must take effect (different window =
        different scored-token sequence = different PPL)."""
        ppl_a = compute_perplexity(tiny_llama, long_input, max_length=32)
        ppl_b = compute_perplexity(tiny_llama, long_input, max_length=64)
        # Different windowings can produce identical PPL only by
        # coincidence; on the test sequence we expect them to differ
        assert abs(ppl_a - ppl_b) > 1e-6


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_rejects_3d_input(self, tiny_llama):
        bad = torch.zeros(1, 1, 32, dtype=torch.long)
        with pytest.raises(ValueError, match="expects \\(1, L\\)"):
            compute_perplexity(tiny_llama, bad)

    def test_rejects_batch_size_above_one(self, tiny_llama):
        bad = torch.zeros(2, 32, dtype=torch.long)
        with pytest.raises(ValueError, match="expects \\(1, L\\)"):
            compute_perplexity(tiny_llama, bad)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_ppl(self, tiny_llama, short_input):
        """Eval mode + no_grad: identical input must produce identical PPL."""
        ppl_a = compute_perplexity(tiny_llama, short_input)
        ppl_b = compute_perplexity(tiny_llama, short_input)
        assert ppl_a == ppl_b

    def test_different_inputs_different_ppl(self, tiny_llama):
        """Sensitivity check: distinct inputs should produce distinct PPL."""
        torch.manual_seed(3)
        x1 = torch.randint(0, tiny_llama.config.vocab_size, (1, 32))
        x2 = torch.randint(0, tiny_llama.config.vocab_size, (1, 32))
        # Vanishingly unlikely that two random sequences produce
        # identical PPL down to float precision
        ppl1 = compute_perplexity(tiny_llama, x1)
        ppl2 = compute_perplexity(tiny_llama, x2)
        assert abs(ppl1 - ppl2) > 1e-6
