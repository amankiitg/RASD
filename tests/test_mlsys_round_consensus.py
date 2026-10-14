"""Round-level consensus and cache-truth regressions (R3, R4).

Two defects that only a multi-rank GPU run would otherwise expose:

* **R4** — the draft TOKENS were never broadcast. Every rank runs the draft
  model, so under the greedy contract its tokens agree and the omission is
  invisible; at temperature > 0 each rank samples its own proposal, so sharing
  the logits is not enough to make `n_acc` agree, and a disagreement there ends
  as a ring P2P size mismatch (the historical SeqNum ~3500 coalesced timeout).

* **R3** — `kv_len_after` was `prior_target_len + committed`, i.e. the
  arithmetic the truncation had been asked to perform. The cap smoke asserts on
  that field, so the assertion restated the code being checked. It is now read
  back from the cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.models.rasd_inference import (
    _broadcast_round_state, _kv_seq_len, _truncate_kv,
)

REPO = Path(__file__).resolve().parent.parent


class _RecordingDist:
    """Stands in for torch.distributed inside the engine module."""

    def __init__(self):
        self.calls: list = []

    def broadcast(self, tensor, src=0):
        self.calls.append((id(tensor), src))


def _state():
    return (torch.zeros(1, 4, dtype=torch.long),
            torch.zeros(1, 5, 8),
            torch.zeros(1, 4, 8))


def test_the_round_state_broadcast_includes_the_draft_tokens(monkeypatch):
    import src.models.rasd_inference as ri
    fake = _RecordingDist()
    monkeypatch.setattr(ri, "dist", fake)
    draft_seq, target_logits_v, draft_logits = _state()

    _broadcast_round_state(draft_seq, target_logits_v, draft_logits, 8)

    broadcast_ids = [i for i, _ in fake.calls]
    assert broadcast_ids == [id(draft_seq), id(target_logits_v),
                             id(draft_logits)], (
        "the draft tokens must be broadcast with the logits: at temperature > 0 "
        "each rank samples its own proposal and n_acc disagrees")
    assert {src for _, src in fake.calls} == {0}


def test_a_single_rank_broadcasts_nothing(monkeypatch):
    """world_size == 1 has no rank 0 to agree with."""
    import src.models.rasd_inference as ri
    fake = _RecordingDist()
    monkeypatch.setattr(ri, "dist", fake)

    _broadcast_round_state(*_state(), 1)

    assert fake.calls == []


class _LyingCache:
    """A cache whose truncate() does not do what it was asked.

    If the round report recomputed the length instead of reading the cache, this
    stand-in could not be detected -- which is the whole point of R3.
    """

    def __init__(self, held: int):
        self._held = held
        self.asked_for = None

    def truncate(self, new_len):
        self.asked_for = new_len
        return self

    def get_seq_length(self, layer_idx: int = 0):
        return self._held


def test_kv_len_is_read_back_from_the_cache():
    cache = _LyingCache(held=1234)
    _truncate_kv(cache, 999)                       # NF4 path: in-place truncate
    assert cache.asked_for == 999

    assert _kv_seq_len(cache) == 1234, (
        "the length must come from the cache, so a truncation that did not "
        "apply is visible rather than restated")


def test_kv_len_from_a_legacy_tuple_and_from_nothing():
    k = torch.zeros(2, 3, 17, 4)
    v = torch.zeros(2, 3, 17, 4)
    assert _kv_seq_len(((k, v),)) == 17
    assert _kv_seq_len(None) == 0


def test_the_real_nf4_cache_reports_its_truncated_length():
    """The stand-in above must not be the only evidence."""
    from src.models.nf4_dynamic_cache import NF4DynamicCache
    cache = NF4DynamicCache(block_size=4, dtype=torch.float32,
                            bf16_prefix_size=0, update_chunk_size=0)
    k = torch.randn(1, 2, 20, 4)
    v = torch.randn(1, 2, 20, 4)
    cache.update(k, v, layer_idx=0)
    assert _kv_seq_len(cache) == 20

    _truncate_kv(cache, 7)
    assert _kv_seq_len(cache) == 7, (
        "NF4DynamicCache.truncate must be honoured and measurable")


def test_the_round_report_does_not_recompute_the_kv_length():
    """The field's source, not its value: it must not be the request again."""
    src = (REPO / "src" / "models" / "rasd_inference.py").read_text()
    assert 'rec["kv_len_after"] = int(kv_len_after)' in src
    assert "kv_len_after = _kv_seq_len(past_kv)" in src
    # The recomputation that R3 removed must not come back.
    assert 'rec["kv_len_after"] = int(prior_target_len + committed)' not in src, (
        "kv_len_after is again the arithmetic the truncation was asked to "
        "perform, so the cap smoke's KV assertion restates the code")
