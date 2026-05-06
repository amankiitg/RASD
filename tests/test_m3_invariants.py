"""Regression tests for the four M3 fixes landed 2026-05-06.

Locks in the architectural and verify-loop invariants from:
  * Option B  (eb9297a) — draft model is built without RoPE scaling
  * Fix2      (e875f6d) — cross-rank logits broadcast before accept/reject
  * Fix3      (45b2b40) — _prefill auto-truncates prompt to multiple of W
  * Fix4      (ad2bf5e) — legacy _ring_peer_loop master/slave is dead code

If any of these regress (silently removed by a refactor, accidentally
disabled, etc.), these tests fail loudly. Mostly source-inspection: the
runtime paths require multi-GPU NCCL init to exercise end-to-end, but
the textual fingerprints are stable.

One functional test (Option B build_hf_config) does mock-instantiate the
config builder to verify the runtime branch actually skips RoPE scaling.
"""

import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

# Match the sys.path convention used by the rest of the test suite
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = Path(__file__).resolve().parent.parent
RASD_INF_SRC = (REPO_ROOT / "src" / "models" / "rasd_inference.py").read_text()
RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()


# ---------------------------------------------------------------------------
# Option B — eb9297a — draft built without RoPE scaling
# ---------------------------------------------------------------------------

class TestOptionB:
    """The draft model must NOT be RoPE-scaled to the target's context.

    Scaling would inflate replicated draft KV from ~770 MB to ~12 GB at
    ctx=64k×W=8, breaking the 40 GB SXM2 memory budget.
    """

    def test_build_hf_config_accepts_apply_rope_scaling(self):
        """Signature regression: kwarg must exist for caller to disable scaling."""
        sig = re.search(
            r"def _build_hf_config\([^)]*apply_rope_scaling[^)]*\)",
            RASD_INF_SRC, re.DOTALL,
        )
        assert sig, (
            "Option B regression: _build_hf_config lost the apply_rope_scaling "
            "parameter — see commit eb9297a"
        )

    def test_draft_call_site_disables_scaling(self):
        """Wherever the draft is loaded, apply_rope_scaling=False must be passed."""
        draft_calls = re.findall(
            r"_build_hf_config\([^)]*label\s*=\s*[\"']draft[\"'][^)]*\)",
            RASD_INF_SRC, re.DOTALL,
        )
        assert draft_calls, "No _build_hf_config draft call site found"
        for call in draft_calls:
            assert "apply_rope_scaling=False" in call, (
                f"Option B regression: draft call missing "
                f"apply_rope_scaling=False:\n{call}"
            )

    def test_draft_keeps_native_max_position(self):
        """Functional: draft branch leaves max_position_embeddings unchanged."""
        from transformers import LlamaConfig
        from src.models.rasd_inference import RASDInference

        fake_cfg = LlamaConfig(max_position_embeddings=4096)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg):
            inst = RASDInference.__new__(RASDInference)  # skip heavy __init__
            result = inst._build_hf_config(
                model_name="ignored",
                revision=None,
                context_length=65536,
                label="draft",
                apply_rope_scaling=False,
            )
        # Some transformers versions populate rope_scaling with a default
        # dict (e.g. {"rope_theta": ..., "rope_type": "default"}). The
        # Option B branch must leave whatever was there untouched —
        # specifically, it must NOT insert `type="linear"` or `factor=...`.
        rs = getattr(result, "rope_scaling", None) or {}
        assert rs.get("type") != "linear", (
            f"Option B regression: draft got linear RoPE scaling applied: {rs}"
        )
        assert "factor" not in rs, (
            f"Option B regression: draft got a scaling factor: {rs}"
        )
        assert result.max_position_embeddings == 4096, (
            "Option B regression: draft max_position_embeddings was modified"
        )

    def test_target_branch_still_scales(self):
        """Sanity: target path (apply_rope_scaling=True) still scales correctly."""
        from transformers import LlamaConfig
        from src.models.rasd_inference import RASDInference

        fake_cfg = LlamaConfig(max_position_embeddings=4096)
        with patch("transformers.AutoConfig.from_pretrained", return_value=fake_cfg):
            inst = RASDInference.__new__(RASDInference)
            result = inst._build_hf_config(
                model_name="ignored",
                revision=None,
                context_length=65536,
                label="target",
                apply_rope_scaling=True,
            )
        assert result.rope_scaling == {"type": "linear", "factor": 16.0}, (
            "RoPE scaling on target broke — Option B target path regression"
        )
        assert result.max_position_embeddings == 65536


# ---------------------------------------------------------------------------
# Fix2 — e875f6d — cross-rank logits broadcast
# ---------------------------------------------------------------------------

class TestFix2:
    """target_logits_v + draft_logits must be broadcast from rank 0 before
    accept/reject. Without this, bf16 numerical drift in the ring online-
    softmax causes ranks to make different acceptance decisions, leading
    to KV-size desync and NCCL coalesced-op timeouts at SeqNum ~3500+.
    """

    def test_target_logits_broadcast_present(self):
        assert re.search(
            r"dist\.broadcast\(\s*target_logits_v\s*,\s*src\s*=\s*0\s*\)",
            RASD_INF_SRC,
        ), (
            "Fix2 regression: dist.broadcast(target_logits_v, src=0) missing — "
            "see commit e875f6d"
        )

    def test_draft_logits_broadcast_present(self):
        assert re.search(
            r"dist\.broadcast\(\s*draft_logits\s*,\s*src\s*=\s*0\s*\)",
            RASD_INF_SRC,
        ), (
            "Fix2 regression: dist.broadcast(draft_logits, src=0) missing — "
            "see commit e875f6d"
        )

    def test_broadcasts_inside_world_size_guard(self):
        """Both broadcasts must be inside `if self._world_size > 1:` so
        single-rank runs stay byte-identical to pre-Fix2."""
        lines = RASD_INF_SRC.splitlines()
        for needle in ("dist.broadcast(target_logits_v",
                       "dist.broadcast(draft_logits"):
            target_idx = next(
                (i for i, ln in enumerate(lines) if needle in ln), None
            )
            assert target_idx is not None, f"could not locate {needle!r}"
            # walk backwards up to 30 lines for the guard
            found = False
            for i in range(target_idx - 1, max(0, target_idx - 30), -1):
                if re.match(r"\s*if self\._world_size > 1:", lines[i]):
                    found = True
                    break
                if re.match(r"\s*def ", lines[i]):
                    break  # left enclosing function without finding guard
            assert found, (
                f"Fix2 regression: {needle!r} is not inside an "
                f"`if self._world_size > 1:` guard"
            )


# ---------------------------------------------------------------------------
# Fix3 — 45b2b40 — _prefill auto-truncates prompt to multiple of world_size
# ---------------------------------------------------------------------------

class TestFix3:
    """Tokenizers regularly return off-by-a-few token counts; the
    contiguous-shard divisibility assertion was crashing rank 0 on
    realistic prompts. Fix3 truncates input_ids to floor(S/W)*W.
    """

    def test_divisibility_guard_present(self):
        assert "S % self._world_size != 0" in RASD_INF_SRC, (
            "Fix3 regression: divisibility guard removed from _prefill — "
            "see commit 45b2b40"
        )

    def test_alignment_math_present(self):
        assert "S_aligned = (S // self._world_size) * self._world_size" in RASD_INF_SRC, (
            "Fix3 regression: alignment math removed from _prefill — "
            "see commit 45b2b40"
        )

    def test_input_ids_sliced(self):
        """Both input_ids and attention_mask (when present) must be sliced."""
        assert "input_ids = input_ids[:, :S_aligned].contiguous()" in RASD_INF_SRC, (
            "Fix3 regression: input_ids not sliced after alignment"
        )
        assert "attention_mask = attention_mask[:, :S_aligned].contiguous()" in RASD_INF_SRC, (
            "Fix3 regression: attention_mask not sliced after alignment"
        )

    def test_truncation_math_invariants(self):
        """Pure math: aligned length is the largest multiple of W <= S."""
        for S, W in [(62660, 8), (65536, 8), (1024, 4),
                     (1023, 4), (16, 8), (17, 8), (8, 8)]:
            S_aligned = (S // W) * W
            assert S_aligned % W == 0, f"misaligned for S={S} W={W}"
            assert S_aligned <= S, f"truncation grew S for S={S} W={W}"
            assert S - S_aligned < W, f"truncation lost too much for S={S} W={W}"


# ---------------------------------------------------------------------------
# Fix4 — ad2bf5e — _ring_peer_loop is dead code, never invoked
# ---------------------------------------------------------------------------

class TestFix4:
    """The pre-R3 master/slave pattern (rank 0 driver, ranks 1..N-1 in
    `dist.recv(tick, src=0)` polling loop) was deleted as a call site.
    The function definition stays as documentation, but invoking it again
    would deadlock all non-zero ranks.
    """

    def test_peer_loop_not_invoked(self):
        invocations = []
        for lineno, line in enumerate(RUN_EXP_SRC.splitlines(), start=1):
            if "_ring_peer_loop(" in line and "def _ring_peer_loop" not in line:
                invocations.append((lineno, line.strip()))
        assert not invocations, (
            "Fix4 regression: _ring_peer_loop is being invoked — see commit "
            f"ad2bf5e. Found at: {invocations}"
        )

    def test_run_single_worker_lockstep_comment_present(self):
        """Defensive: the docstring explaining 'all ranks lockstep' should
        remain so the next reader knows not to reintroduce master/slave."""
        assert "ALL ranks run the full" in RUN_EXP_SRC, (
            "Fix4 regression: lockstep documentation deleted from "
            "_run_single_worker — risk of reintroducing master/slave"
        )
