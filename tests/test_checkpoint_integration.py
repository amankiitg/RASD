"""Tests for the C6 wiring of checkpoint/resume into generate().

The full generate() flow needs CUDA + actual model weights to exercise
end-to-end. Locally we test:

1. Default-off invariant: cfg.checkpoint_every == 0 produces the M3
   byte-identical execution path (gate is correctly placed)
2. RASDConfig defaults — checkpoint_every=0, checkpoint_dir=None,
   run_id=None
3. _try_load_checkpoint() returns None when config is unset
4. _maybe_save_checkpoint is a no-op when cfg.checkpoint_every == 0,
   even when other fields are set
5. Source-inspection that the resume gate and save call are placed
   correctly in generate()
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from src.models.rasd_inference import RASDConfig, RASDInference

REPO_ROOT = Path(__file__).resolve().parent.parent
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()


# ---------------------------------------------------------------------------
# Config defaults — M3 byte-identical invariant
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_checkpoint_every_default_zero(self):
        """C6 must default to disabled so M3 replay is byte-identical."""
        cfg = RASDConfig()
        assert cfg.checkpoint_every == 0

    def test_checkpoint_dir_default_none(self):
        cfg = RASDConfig()
        assert cfg.checkpoint_dir is None

    def test_run_id_default_none(self):
        cfg = RASDConfig()
        assert cfg.run_id is None

    def test_can_be_set(self):
        """All three fields are settable when the user opts in."""
        cfg = RASDConfig(
            checkpoint_every=4,
            checkpoint_dir="/tmp/ckpts",
            run_id="A2_k4_s42",
        )
        assert cfg.checkpoint_every == 4
        assert cfg.checkpoint_dir == "/tmp/ckpts"
        assert cfg.run_id == "A2_k4_s42"


# ---------------------------------------------------------------------------
# Helper methods on RASDInference (without booting the engine)
# ---------------------------------------------------------------------------

def _make_bare_engine(cfg, rank: int = 0):
    """A bare RASDInference instance — bypasses __init__ heavy lifting.

    Used to exercise the helper methods (_try_load_checkpoint and
    _maybe_save_checkpoint) without loading models or initializing CUDA.
    """
    inst = RASDInference.__new__(RASDInference)
    inst.cfg = cfg
    inst._rank = rank
    inst._world_size = 1
    inst._device = torch.device("cpu")
    return inst


class TestTryLoadCheckpoint:
    def test_returns_none_when_dir_unset(self):
        cfg = RASDConfig(checkpoint_every=4, run_id="X")  # dir=None
        engine = _make_bare_engine(cfg)
        assert engine._try_load_checkpoint() is None

    def test_returns_none_when_run_id_unset(self, tmp_path):
        cfg = RASDConfig(checkpoint_every=4, checkpoint_dir=str(tmp_path))
        engine = _make_bare_engine(cfg)
        assert engine._try_load_checkpoint() is None

    def test_returns_none_when_no_files_present(self, tmp_path):
        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path),
                         run_id="A2_k4_s42")
        engine = _make_bare_engine(cfg)
        assert engine._try_load_checkpoint() is None

    def test_loads_existing_checkpoint(self, tmp_path):
        """When a checkpoint exists, _try_load_checkpoint returns it."""
        from src.models.checkpoint import GenerationCheckpoint, checkpoint_path

        ckpt = GenerationCheckpoint(
            n_rounds=4, global_seqlen=128,
            total_accepted=10, total_draft_toks=16, prefill_len=120,
            cur_token=torch.tensor([[42]]),
            generated=[torch.tensor([[1]]), torch.tensor([[2]])],
            past_kv=tuple([
                (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            ]),
            draft_past_kv=tuple([
                (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            ]),
        )
        path = checkpoint_path(tmp_path, "X", round_idx=4, rank=0)
        ckpt.save(path)

        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path), run_id="X")
        engine = _make_bare_engine(cfg)
        loaded = engine._try_load_checkpoint()
        assert loaded is not None
        assert loaded.n_rounds == 4
        assert loaded.global_seqlen == 128


class TestMaybeSaveCheckpoint:
    def _kw(self):
        """Stock kwargs for _maybe_save_checkpoint — small fake state."""
        return dict(
            n_rounds=4, global_seqlen=128,
            total_accepted=10, total_draft_toks=16,
            cur_token=torch.tensor([[42]]),
            generated=[torch.tensor([[1]]), torch.tensor([[2]])],
            past_kv=tuple([
                (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            ]),
            draft_past_kv=tuple([
                (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            ]),
            per_token_trace=[],
            prefill_len=120,
        )

    def test_noop_when_disabled(self, tmp_path):
        """checkpoint_every == 0 -> no file written, even with dir + run_id set."""
        cfg = RASDConfig(checkpoint_every=0,
                         checkpoint_dir=str(tmp_path), run_id="X")
        engine = _make_bare_engine(cfg)
        engine._maybe_save_checkpoint(**self._kw())
        assert list(tmp_path.iterdir()) == []

    def test_noop_when_no_dir(self):
        """checkpoint_dir unset -> no save attempt."""
        cfg = RASDConfig(checkpoint_every=4, run_id="X")  # dir=None
        engine = _make_bare_engine(cfg)
        # Must not raise / not crash
        engine._maybe_save_checkpoint(**self._kw())

    def test_noop_when_no_run_id(self, tmp_path):
        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path))  # run_id=None
        engine = _make_bare_engine(cfg)
        engine._maybe_save_checkpoint(**self._kw())
        assert list(tmp_path.iterdir()) == []

    def test_noop_on_off_round(self, tmp_path):
        """checkpoint_every=4 + n_rounds=3 -> no save (3 % 4 != 0)."""
        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path), run_id="X")
        engine = _make_bare_engine(cfg)
        kw = self._kw()
        kw["n_rounds"] = 3
        engine._maybe_save_checkpoint(**kw)
        # Run dir might not exist if no save; either way no .pt files
        if (tmp_path / "X").exists():
            assert list((tmp_path / "X").iterdir()) == []

    def test_saves_on_scheduled_round(self, tmp_path):
        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path), run_id="X")
        engine = _make_bare_engine(cfg)
        engine._maybe_save_checkpoint(**self._kw())
        files = list((tmp_path / "X").glob("*.pt"))
        assert len(files) == 1
        assert files[0].name == "round_4_rank_0.pt"

    def test_per_rank_filename(self, tmp_path):
        cfg = RASDConfig(checkpoint_every=4,
                         checkpoint_dir=str(tmp_path), run_id="X")
        for rank in (0, 3, 7):
            engine = _make_bare_engine(cfg, rank=rank)
            engine._maybe_save_checkpoint(**self._kw())
        names = sorted(p.name for p in (tmp_path / "X").glob("*.pt"))
        assert names == [
            "round_4_rank_0.pt",
            "round_4_rank_3.pt",
            "round_4_rank_7.pt",
        ]


# ---------------------------------------------------------------------------
# Source-inspection guards on the integration points in generate()
# ---------------------------------------------------------------------------

class TestGenerateWiring:
    def test_resume_gate_present(self):
        """generate() must check for an existing checkpoint at entry,
        gated on cfg.checkpoint_every > 0."""
        assert re.search(
            r"ckpt\s*=\s*self\._try_load_checkpoint\(\)\s*if\s+cfg\.checkpoint_every\s*>\s*0\s+else\s+None",
            RASD_INF_SRC,
        ), (
            "C6 regression: resume gate missing from generate() entry"
        )

    def test_prefill_skipped_on_resume(self):
        """The prefill block must be inside `if ckpt is None:` — otherwise
        we'd run prefill even when restoring from a checkpoint."""
        assert "if ckpt is None:" in RASD_INF_SRC, (
            "C6 regression: `if ckpt is None:` guard missing — prefill "
            "would run unconditionally"
        )

    def test_resume_else_branch_restores_state(self):
        """The else branch of `if ckpt is None:` must restore variables
        from the checkpoint."""
        # Search for the restore lines inside the else branch
        for needle in ("ckpt.past_kv", "ckpt.draft_past_kv", "ckpt.cur_token",
                       "ckpt.n_rounds", "ckpt.global_seqlen",
                       "ckpt.total_accepted", "ckpt.prefill_len"):
            assert needle in RASD_INF_SRC, (
                f"C6 regression: resume branch missing restore of {needle!r}"
            )

    def test_save_called_in_verify_loop(self):
        """_maybe_save_checkpoint must be invoked from inside the verify
        loop. The default-off check is inside the helper itself."""
        assert "self._maybe_save_checkpoint(" in RASD_INF_SRC, (
            "C6 regression: _maybe_save_checkpoint not called from generate()"
        )

    def test_set_prefill_len_called_on_resume(self):
        """On resume under multi-rank, set_prefill_len must be called so
        the patched ring attention knows the prefill boundary."""
        # Find the resume branch and verify set_prefill_len is called there
        idx_resume = RASD_INF_SRC.find("ckpt.prefill_len")
        assert idx_resume > 0
        # Look for set_prefill_len call within the next ~500 chars
        window = RASD_INF_SRC[idx_resume:idx_resume + 500]
        assert "set_prefill_len(self.target_model" in window, (
            "C6 regression: set_prefill_len not called in resume branch"
        )
