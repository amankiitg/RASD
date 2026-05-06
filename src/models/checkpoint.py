"""Generation checkpoint + resume primitives (M4 C6).

At 1M context a single RASD run is 20+ minutes — one crash = one
pod-hour lost without checkpointing. C6 saves verify-loop state
periodically so a crashed run can resume from the last checkpoint
instead of re-running prefill (the bulk of wall-time at long context).

Design contract:
- Default OFF. When `cfg.checkpoint_every == 0` (the default), generate()
  runs the M3-byte-identical path with zero overhead.
- Save side: rank 0 only. After every Nth round, dump verify-loop
  state to `<dir>/<run_id>/round_<n>.pt` via torch.save. Atomic write
  via .tmp + rename so a crash mid-write doesn't corrupt the checkpoint.
- Load side: at generate() entry, if a checkpoint for this run_id
  exists, restore state and skip prefill entirely (the past_kv tensors
  in the checkpoint already encode the prefill).
- Pure save/load primitives in this module; integration into
  generate() is in rasd_inference.py.

Risk reduction: the GenerationCheckpoint dataclass is defined here as
a single, testable unit. The integration point in generate() reads
from this module's API and never touches the on-disk format directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def checkpoint_dir_for_run(base_dir: str | Path, run_id: str) -> Path:
    """`<base_dir>/<run_id>/` — one subdir per run keeps the layout grep-friendly."""
    return Path(base_dir).resolve() / run_id


def checkpoint_path(base_dir: str | Path, run_id: str,
                    round_idx: int, rank: int = 0) -> Path:
    """`<base_dir>/<run_id>/round_<round_idx>_rank_<rank>.pt`.

    Per-rank file because each rank holds its own KV slice under ring
    attention — a single combined file would require an all-gather
    we'd rather avoid in the hot path. Default rank=0 so single-rank
    runs and tests don't need to think about ranks.
    """
    return (checkpoint_dir_for_run(base_dir, run_id)
            / f"round_{round_idx}_rank_{rank}.pt")


def latest_checkpoint(base_dir: str | Path, run_id: str,
                      rank: int = 0) -> Optional[Path]:
    """Find the highest-round checkpoint for this run + rank, or None.

    Sorts by integer round index, not lex-sort, so round_10_rank_0.pt
    beats round_2_rank_0.pt correctly.
    """
    d = checkpoint_dir_for_run(base_dir, run_id)
    if not d.exists():
        return None
    suffix = f"_rank_{rank}.pt"
    candidates: list[tuple[int, Path]] = []
    for p in d.glob(f"round_*_rank_{rank}.pt"):
        stem = p.name
        # Filename: round_<idx>_rank_<rank>.pt
        if not stem.endswith(suffix):
            continue
        try:
            idx_str = stem[len("round_"):-len(suffix)]
            idx = int(idx_str)
        except (ValueError, IndexError):
            continue
        candidates.append((idx, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ---------------------------------------------------------------------------
# Checkpoint payload
# ---------------------------------------------------------------------------

@dataclass
class GenerationCheckpoint:
    """Verify-loop state needed to resume generate() after the prefill point.

    All tensor fields are torch.Tensor (any device — caller moves to the
    right device after load). past_kv and draft_past_kv are tuples-of-
    tuples of (k, v) tensors per layer — the legacy HF format.
    """
    # Counters
    n_rounds:           int
    global_seqlen:      int
    total_accepted:     int
    total_draft_toks:   int
    prefill_len:        int

    # Latest token (B, 1)
    cur_token:          torch.Tensor

    # Concatenable per-step tokens (each (B, 1)); generate() concats
    # input_ids + generated -> generated_ids at the end
    generated:          list  # list[torch.Tensor]

    # KV caches (legacy tuple format). past_kv = target; draft_past_kv = draft.
    past_kv:            tuple
    draft_past_kv:      tuple

    # C13 per-position acceptance trace accumulated so far (may be empty)
    per_token_trace:    list = field(default_factory=list)

    # Random state so resume produces identical continuation given the
    # same seed (best effort — doesn't capture per-rank RNG under
    # multi-rank, but Fix2 broadcasts logits so accept/reject is
    # deterministic across ranks anyway)
    rng_state:          Optional[torch.Tensor] = None

    # ------------------------- I/O -------------------------

    def save(self, path: str | Path) -> Path:
        """Atomic torch.save: write to .tmp then rename. Never leaves a
        half-written file even if the process dies mid-write."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # Serialize as a plain dict so the on-disk format is not coupled
        # to the GenerationCheckpoint class definition.
        payload = {
            "n_rounds":         self.n_rounds,
            "global_seqlen":    self.global_seqlen,
            "total_accepted":   self.total_accepted,
            "total_draft_toks": self.total_draft_toks,
            "prefill_len":      self.prefill_len,
            "cur_token":        self.cur_token,
            "generated":        list(self.generated),
            "past_kv":          self.past_kv,
            "draft_past_kv":    self.draft_past_kv,
            "per_token_trace":  list(self.per_token_trace),
            "rng_state":        self.rng_state,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GenerationCheckpoint":
        """Load a checkpoint produced by `save`.

        weights_only=False: the payload contains nested tuples + lists +
        primitives, all torch-safe. We control both sides of the
        save/load boundary so the broader weights_only=True security
        model isn't a fit here.
        """
        d = torch.load(str(path), weights_only=False, map_location="cpu")
        return cls(
            n_rounds         = int(d["n_rounds"]),
            global_seqlen    = int(d["global_seqlen"]),
            total_accepted   = int(d["total_accepted"]),
            total_draft_toks = int(d["total_draft_toks"]),
            prefill_len      = int(d["prefill_len"]),
            cur_token        = d["cur_token"],
            generated        = list(d["generated"]),
            past_kv          = tuple(d["past_kv"]) if d["past_kv"] is not None else None,
            draft_past_kv    = tuple(d["draft_past_kv"]) if d["draft_past_kv"] is not None else None,
            per_token_trace  = list(d.get("per_token_trace", [])),
            rng_state        = d.get("rng_state"),
        )

    # ------------------------- Restore -------------------------

    def move_tensors_to(self, device: torch.device | str) -> "GenerationCheckpoint":
        """Move every tensor in this checkpoint to `device`. Returns self."""
        device = torch.device(device)
        self.cur_token = self.cur_token.to(device)
        self.generated = [t.to(device) for t in self.generated]
        if self.past_kv is not None:
            self.past_kv = tuple(
                tuple(t.to(device) for t in layer) for layer in self.past_kv
            )
        if self.draft_past_kv is not None:
            self.draft_past_kv = tuple(
                tuple(t.to(device) for t in layer) for layer in self.draft_past_kv
            )
        return self


# ---------------------------------------------------------------------------
# Convenience: should we save this round?
# ---------------------------------------------------------------------------

def should_save_this_round(checkpoint_every: int, n_rounds: int) -> bool:
    """`checkpoint_every == 0` -> never. Otherwise save every Nth round.

    Pure helper so the `if` condition in generate() is locked in by tests.
    """
    if checkpoint_every <= 0:
        return False
    if n_rounds <= 0:
        return False
    return (n_rounds % checkpoint_every) == 0
