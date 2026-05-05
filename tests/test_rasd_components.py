"""
Unit tests for RASD inference components.

Covers the riskiest pieces before running the full ablation grid:
  1. _sample             — token sampling (greedy + stochastic)
  2. _acceptance_mask    — accept/reject criterion (Leviathan et al.)
  3. RASDConfig defaults — guard against accidental knob breakage

The former AsyncKVRingPrefetcher tests were removed in R3 (the prefetcher
itself was deleted when ring rotation moved into the LlamaAttention forward;
see M3_RING_INTEGRATION_PLAN.md). Ring kernel and patch correctness now lives
in tests/test_ring_attention.py and tests/test_ring_llama_attention.py.

Run on the pod:
    cd /workspace/RASD && python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from src.models.rasd_inference import (
    _sample,
    _acceptance_mask,
    RASDConfig,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab_logits():
    """Batch of 2, vocab size 1000."""
    torch.manual_seed(42)
    return torch.randn(2, 1000, device=DEVICE)


@pytest.fixture
def spec_tensors():
    """Draft tokens + logits + target logits for a batch=1, k=4 scenario."""
    torch.manual_seed(42)
    B, k, V = 1, 4, 500
    draft_tokens  = torch.randint(0, V, (B, k), device=DEVICE)
    draft_logits  = torch.randn(B, k, V, device=DEVICE)
    target_logits = torch.randn(B, k + 1, V, device=DEVICE)
    return draft_tokens, draft_logits, target_logits


# ---------------------------------------------------------------------------
# 1. _sample
# ---------------------------------------------------------------------------

class TestSample:

    def test_greedy_returns_argmax(self, vocab_logits):
        """temperature=0 must return the argmax token."""
        out = _sample(vocab_logits, temperature=0.0, top_p=1.0)
        expected = vocab_logits.argmax(dim=-1)
        assert out.shape == (2,)
        assert torch.equal(out, expected)

    def test_output_shape(self, vocab_logits):
        """Output should be (B,) for any valid temperature."""
        out = _sample(vocab_logits, temperature=1.0, top_p=1.0)
        assert out.shape == (2,)

    def test_tokens_in_vocab_range(self, vocab_logits):
        """Sampled tokens must be valid vocab indices."""
        V = vocab_logits.shape[-1]
        for temp in [0.5, 1.0, 2.0]:
            out = _sample(vocab_logits, temperature=temp, top_p=1.0)
            assert (out >= 0).all() and (out < V).all()

    def test_top_p_restricts_to_nucleus(self):
        """With top_p=0.0001 all probability mass on one token → always picks it."""
        logits = torch.full((1, 100), -1e9, device=DEVICE)
        logits[0, 7] = 100.0   # token 7 dominates
        for _ in range(20):
            out = _sample(logits, temperature=1.0, top_p=0.0001)
            assert out.item() == 7, f"Expected 7, got {out.item()}"

    def test_deterministic_with_seed(self, vocab_logits):
        """Same seed → same stochastic output."""
        torch.manual_seed(99)
        out1 = _sample(vocab_logits, temperature=1.0, top_p=0.9)
        torch.manual_seed(99)
        out2 = _sample(vocab_logits, temperature=1.0, top_p=0.9)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# 2. _acceptance_mask
# ---------------------------------------------------------------------------

class TestAcceptanceMask:

    def test_shape(self, spec_tensors):
        draft_tokens, draft_logits, target_logits = spec_tensors
        accepted, n_acc = _acceptance_mask(draft_tokens, target_logits, draft_logits, temperature=1.0)
        assert accepted.shape == draft_tokens.shape
        assert 0 <= n_acc <= draft_tokens.shape[1]

    def test_greedy_accepts_matching_tokens(self):
        """Greedy (temp=0): accept iff target argmax == draft token."""
        B, k, V = 1, 4, 100
        # Force target logits so argmax at positions 0,1 match draft, 2 doesn't
        target_logits = torch.zeros(B, k + 1, V, device=DEVICE)
        draft_tokens  = torch.tensor([[5, 10, 15, 20]], device=DEVICE)
        # positions 0,1: target argmax = draft token
        target_logits[0, 0, 5]  = 10.0
        target_logits[0, 1, 10] = 10.0
        # position 2: target argmax ≠ draft token
        target_logits[0, 2, 99] = 10.0   # draft said 15, target says 99
        target_logits[0, 3, 20] = 10.0
        draft_logits = torch.zeros(B, k, V, device=DEVICE)

        accepted, n_acc = _acceptance_mask(draft_tokens, target_logits, draft_logits, temperature=0.0)
        assert accepted[0, 0].item() is True
        assert accepted[0, 1].item() is True
        assert accepted[0, 2].item() is False
        assert n_acc == 2

    def test_n_acc_is_first_rejection(self, spec_tensors):
        """n_acc should be the index of the first False in accepted[0]."""
        draft_tokens, draft_logits, target_logits = spec_tensors
        accepted, n_acc = _acceptance_mask(draft_tokens, target_logits, draft_logits, temperature=1.0)
        first_false = (accepted[0] == False).nonzero(as_tuple=False)
        expected_n  = first_false[0].item() if len(first_false) > 0 else draft_tokens.shape[1]
        assert n_acc == expected_n

    def test_all_accepted_when_target_equals_draft(self):
        """If target distribution == draft distribution, accept prob = 1 for all."""
        B, k, V = 1, 4, 50
        shared_logits = torch.randn(B, k, V, device=DEVICE)
        draft_tokens  = shared_logits.argmax(dim=-1)          # greedy draft picks

        # target logits for positions 0..k-1 match draft; position k is bonus
        target_logits = torch.cat([shared_logits,
                                    torch.randn(B, 1, V, device=DEVICE)], dim=1)

        accepted, n_acc = _acceptance_mask(draft_tokens, target_logits, shared_logits, temperature=0.0)
        assert accepted.all(), "All tokens should be accepted when distributions match"
        assert n_acc == k

    def test_acceptance_rate_bounded(self):
        """Acceptance rate must always be in [0, 1]."""
        torch.manual_seed(0)
        for _ in range(10):
            B, k, V = 1, 6, 200
            dt = torch.randint(0, V, (B, k), device=DEVICE)
            dl = torch.randn(B, k, V, device=DEVICE)
            tl = torch.randn(B, k + 1, V, device=DEVICE)
            _, n_acc = _acceptance_mask(dt, tl, dl, temperature=1.0)
            assert 0 <= n_acc <= k


# ---------------------------------------------------------------------------
# 3. RASDConfig defaults
# ---------------------------------------------------------------------------

class TestRASDConfig:

    def test_defaults_are_valid(self):
        cfg = RASDConfig()
        assert cfg.spec_steps > 0
        assert cfg.kv_block_size > 0
        assert cfg.prefetch_depth in (0, 1, 2)
        assert cfg.torch_dtype in (torch.float16, torch.bfloat16)

    def test_bfloat16_dtype(self):
        cfg = RASDConfig(dtype="bfloat16")
        assert cfg.torch_dtype == torch.bfloat16

    def test_float16_dtype(self):
        cfg = RASDConfig(dtype="float16")
        assert cfg.torch_dtype == torch.float16

    def test_all_ablation_configs_valid(self):
        """Every (level × seed) combination from the milestone spec should construct."""
        ablation_configs = [
            # A1
            dict(draft_model_name="distilgpt2"),
            dict(draft_model_name="princeton-nlp/Sheared-LLaMA-1.3B"),
            # A2
            dict(spec_steps=2), dict(spec_steps=4), dict(spec_steps=6),
            dict(spec_steps=8), dict(spec_steps=12),
            # A3
            dict(kv_block_size=256), dict(kv_block_size=512),
            dict(kv_block_size=1024), dict(kv_block_size=2048),
            # A4
            dict(prefetch_depth=0), dict(prefetch_depth=1), dict(prefetch_depth=2),
            # A5
            dict(target_model_name="meta-llama/Llama-2-7b-hf"),
            dict(target_model_name="mistralai/Mistral-7B-v0.1"),
        ]
        for seed in [42, 123, 456]:
            for overrides in ablation_configs:
                cfg = RASDConfig(seed=seed, **overrides)
                assert cfg.seed == seed
