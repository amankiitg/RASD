"""Tests for src/models/checkpoint.py — the M4 C6 save/load primitives.

CPU-only. Save/load round-trip is the whole game; the integration into
generate() is source-inspected separately.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.checkpoint import (
    GenerationCheckpoint,
    checkpoint_dir_for_run,
    checkpoint_path,
    latest_checkpoint,
    should_save_this_round,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

class TestPaths:
    def test_dir_per_run(self, tmp_path):
        d = checkpoint_dir_for_run(tmp_path, "A2_k4_s42")
        assert d.parent == tmp_path.resolve()
        assert d.name == "A2_k4_s42"

    def test_path_includes_round_and_rank(self, tmp_path):
        p = checkpoint_path(tmp_path, "X", round_idx=7)
        assert p.name == "round_7_rank_0.pt"
        assert p.parent.name == "X"

    def test_rank_kwarg_changes_filename(self, tmp_path):
        a = checkpoint_path(tmp_path, "X", round_idx=7, rank=0)
        b = checkpoint_path(tmp_path, "X", round_idx=7, rank=3)
        assert a.name == "round_7_rank_0.pt"
        assert b.name == "round_7_rank_3.pt"
        assert a != b

    def test_distinct_runs_distinct_dirs(self, tmp_path):
        a = checkpoint_dir_for_run(tmp_path, "A")
        b = checkpoint_dir_for_run(tmp_path, "B")
        assert a != b


class TestLatestCheckpoint:
    def test_no_dir_returns_none(self, tmp_path):
        assert latest_checkpoint(tmp_path, "missing_run") is None

    def test_empty_dir_returns_none(self, tmp_path):
        d = checkpoint_dir_for_run(tmp_path, "X")
        d.mkdir(parents=True)
        assert latest_checkpoint(tmp_path, "X") is None

    def test_returns_highest_round(self, tmp_path):
        d = checkpoint_dir_for_run(tmp_path, "X")
        d.mkdir(parents=True)
        for n in (1, 5, 10, 2):
            (d / f"round_{n}_rank_0.pt").write_bytes(b"x")
        result = latest_checkpoint(tmp_path, "X")
        assert result is not None
        assert result.name == "round_10_rank_0.pt"

    def test_integer_sort_not_lex(self, tmp_path):
        """round_10 must beat round_2 — not lex-sorted."""
        d = checkpoint_dir_for_run(tmp_path, "X")
        d.mkdir(parents=True)
        for n in (2, 10):
            (d / f"round_{n}_rank_0.pt").write_bytes(b"x")
        assert latest_checkpoint(tmp_path, "X").name == "round_10_rank_0.pt"

    def test_ignores_malformed_filenames(self, tmp_path):
        d = checkpoint_dir_for_run(tmp_path, "X")
        d.mkdir(parents=True)
        (d / "round_5_rank_0.pt").write_bytes(b"x")
        (d / "round_garbage_rank_0.pt").write_bytes(b"x")
        (d / "not_a_round.pt").write_bytes(b"x")
        assert latest_checkpoint(tmp_path, "X").name == "round_5_rank_0.pt"

    def test_per_rank_isolation(self, tmp_path):
        """rank=1 must not see rank=0's checkpoints (they hold different
        KV slices under ring attention)."""
        d = checkpoint_dir_for_run(tmp_path, "X")
        d.mkdir(parents=True)
        (d / "round_5_rank_0.pt").write_bytes(b"x")
        (d / "round_3_rank_1.pt").write_bytes(b"x")
        assert latest_checkpoint(tmp_path, "X", rank=0).name == "round_5_rank_0.pt"
        assert latest_checkpoint(tmp_path, "X", rank=1).name == "round_3_rank_1.pt"


# ---------------------------------------------------------------------------
# should_save_this_round
# ---------------------------------------------------------------------------

class TestShouldSaveThisRound:
    def test_zero_disables(self):
        for n in (0, 1, 5, 100):
            assert should_save_this_round(0, n) is False

    def test_negative_disables(self):
        assert should_save_this_round(-1, 5) is False

    def test_round_zero_never_saves(self):
        """We checkpoint AFTER a round, so round 0 (pre-loop) shouldn't save."""
        assert should_save_this_round(4, 0) is False

    def test_modulo_save(self):
        assert should_save_this_round(4, 4) is True
        assert should_save_this_round(4, 8) is True
        assert should_save_this_round(4, 12) is True
        assert should_save_this_round(4, 5) is False
        assert should_save_this_round(4, 7) is False

    def test_every_round(self):
        assert should_save_this_round(1, 1) is True
        assert should_save_this_round(1, 2) is True


# ---------------------------------------------------------------------------
# Save/load round-trip
# ---------------------------------------------------------------------------

def _make_fake_kv(num_layers=2, B=1, H=2, S=8, D=4):
    """Tuple-of-tuples (k, v) per layer, matching HF legacy past_kv format."""
    return tuple(
        (torch.randn(B, H, S, D), torch.randn(B, H, S, D))
        for _ in range(num_layers)
    )


@pytest.fixture
def fake_checkpoint():
    torch.manual_seed(0)
    return GenerationCheckpoint(
        n_rounds=4,
        global_seqlen=128,
        total_accepted=10,
        total_draft_toks=16,
        prefill_len=120,
        cur_token=torch.tensor([[42]]),
        generated=[torch.tensor([[10]]), torch.tensor([[20]]),
                   torch.tensor([[30]]), torch.tensor([[40]])],
        past_kv=_make_fake_kv(num_layers=3, S=128),
        draft_past_kv=_make_fake_kv(num_layers=2, S=128),
        per_token_trace=[
            {"round_idx": 0, "global_pos_start": 100, "spec_steps": 4,
             "n_acc": 2, "draft_tokens": [1, 2, 3, 4],
             "accepted": [True, True, False, False]},
        ],
    )


class TestSaveLoadRoundTrip:
    def test_basic_round_trip(self, tmp_path, fake_checkpoint):
        path = tmp_path / "ckpt.pt"
        fake_checkpoint.save(path)
        assert path.exists()
        loaded = GenerationCheckpoint.load(path)

        assert loaded.n_rounds == fake_checkpoint.n_rounds
        assert loaded.global_seqlen == fake_checkpoint.global_seqlen
        assert loaded.total_accepted == fake_checkpoint.total_accepted
        assert loaded.total_draft_toks == fake_checkpoint.total_draft_toks
        assert loaded.prefill_len == fake_checkpoint.prefill_len
        assert torch.equal(loaded.cur_token, fake_checkpoint.cur_token)
        assert len(loaded.generated) == len(fake_checkpoint.generated)
        for a, b in zip(loaded.generated, fake_checkpoint.generated):
            assert torch.equal(a, b)
        assert loaded.per_token_trace == fake_checkpoint.per_token_trace

    def test_kv_tensors_round_trip_exactly(self, tmp_path, fake_checkpoint):
        path = tmp_path / "ckpt.pt"
        fake_checkpoint.save(path)
        loaded = GenerationCheckpoint.load(path)
        assert len(loaded.past_kv) == len(fake_checkpoint.past_kv)
        for la, lb in zip(loaded.past_kv, fake_checkpoint.past_kv):
            assert torch.equal(la[0], lb[0])
            assert torch.equal(la[1], lb[1])

    def test_atomic_save_no_tmp_left_on_disk(self, tmp_path, fake_checkpoint):
        """save() writes to .tmp + os.replace; after success no .tmp remains."""
        path = tmp_path / "ckpt.pt"
        fake_checkpoint.save(path)
        assert path.exists()
        assert not path.with_suffix(path.suffix + ".tmp").exists()

    def test_save_creates_parent_dir(self, tmp_path, fake_checkpoint):
        path = tmp_path / "deep" / "nested" / "ckpt.pt"
        fake_checkpoint.save(path)
        assert path.exists()

    def test_save_overwrites_existing(self, tmp_path, fake_checkpoint):
        """Subsequent saves to the same path replace the old file."""
        path = tmp_path / "ckpt.pt"
        fake_checkpoint.save(path)
        first = path.read_bytes()
        # Modify and re-save
        fake_checkpoint.n_rounds = 999
        fake_checkpoint.save(path)
        loaded = GenerationCheckpoint.load(path)
        assert loaded.n_rounds == 999

    def test_empty_per_token_trace_ok(self, tmp_path):
        """When C13 trace logging is off, per_token_trace is empty.
        Round-trip must work."""
        ckpt = GenerationCheckpoint(
            n_rounds=2, global_seqlen=10, total_accepted=4,
            total_draft_toks=8, prefill_len=8,
            cur_token=torch.tensor([[1]]),
            generated=[torch.tensor([[1]]), torch.tensor([[2]])],
            past_kv=_make_fake_kv(),
            draft_past_kv=_make_fake_kv(),
        )
        path = tmp_path / "ckpt.pt"
        ckpt.save(path)
        loaded = GenerationCheckpoint.load(path)
        assert loaded.per_token_trace == []


# ---------------------------------------------------------------------------
# Tensor device move
# ---------------------------------------------------------------------------

class TestMoveTensorsTo:
    def test_move_to_cpu_is_noop_for_cpu_tensors(self, fake_checkpoint):
        """No-op when already on CPU; doesn't crash."""
        fake_checkpoint.move_tensors_to("cpu")
        assert fake_checkpoint.cur_token.device.type == "cpu"

    def test_move_returns_self(self, fake_checkpoint):
        """move_tensors_to returns self so it chains: load(...).move_tensors_to(...)."""
        result = fake_checkpoint.move_tensors_to("cpu")
        assert result is fake_checkpoint

    def test_handles_none_kv(self, fake_checkpoint):
        """If past_kv is None (e.g., very early checkpoint), move_tensors_to
        must not crash."""
        fake_checkpoint.past_kv = None
        fake_checkpoint.move_tensors_to("cpu")
        assert fake_checkpoint.past_kv is None
