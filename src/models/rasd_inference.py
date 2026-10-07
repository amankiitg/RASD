"""
RASD Inference — Ring Attention Speculative Decoding

Architecture (post-R3 dual-cache integration, 2026-05-05)
---------------------------------------------------------
Two CUDA streams run concurrently at each decoding step:

  stream_compute  — target model verification forward pass
                    (ring attention happens INSIDE this forward, layer by layer,
                    not as a separate prefetch loop in generate())
  stream_draft    — draft model token generation (k steps)

Pipeline per step
-----------------
  [sync]    stream_draft and stream_compute wait for default stream's
            commits from previous round (cur_token, past_kv, draft_past_kv
            after _truncate_kv). Cheap event-based waits.
  [draft]   generate k draft tokens         (stream_draft)
  [verify]  target packed forward over [cur_token, draft_seq]
            with ring attention rotating sharded prefill K/V across ranks
            internally per layer; replicated tail handled locally.
                                            (stream_compute)
  [accept]  accept/reject + bonus sample    (default stream)

Stream-ordering invariants (post-R3 + R5 audit, 2026-05-05)
----------------------------------------------------------
1. Top of each loop iter: stream_draft and stream_compute wait for
   default stream so they observe last round's bonus_sample + truncation.
2. stream_compute.wait_stream(stream_draft) before verify forward — verify
   reads draft_seq written on stream_draft.
3. The torch.cat / torch.stack that build draft_seq and draft_logits must
   live INSIDE the `with stream_draft:` block so the resulting tensors
   stay on stream_draft (not the default stream).
4. torch.cuda.current_stream().wait_stream(stream_compute) after verify —
   accept/reject, bonus sample, _truncate_kv on default stream read
   target_logits_v + post_verify_kv produced on stream_compute.

KV cache layout under multi-rank (R0.1, contiguous):
  Rank r holds prefill positions [r*S/W, (r+1)*S/W) sharded contiguously.
  During decode, every rank also appends new K/V positions to a "replicated
  tail" (identical on all ranks, since hidden_states is replicated).
  Ring rotates only the sharded prefill; tail is local to each rank.

Ablation hooks
---------------
  A1  draft_model_name   TinyLlama-1.1B | Sheared-LLaMA-1.3B
  A2  spec_steps k       2 | 4 | 6 | 8 | 12
  A3  kv_block_size      256 | 512 | 1024 | 2048 (ring transmission chunk size)
  A4  prefetch_depth     0 (sync) | 1 (async-1) | 2 (async-2 — saturates as 1)
  A5  target_model_name  Llama-2-7b-hf

A3/A4 redefined in R3.5 (post-prefetcher-removal): A3 is now the per-step
batch_isend_irecv chunk size inside the ring (smaller = more launch
overhead, larger = better bandwidth amortization). A4 is the ring-step
prefetch depth (whether rotation s+1 is issued before computing step s).
See docs/dev/M3_RING_INTEGRATION_PLAN.md open-question 1a for rationale.

Debug mode
----------
Set debug=True to force synchronisation after every stream operation
and emit verbose per-step logs. Use this to catch race conditions
before running at scale.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NVTX phase markers (M4 Phase C 2026-05-10, codex review on p36 profiler)
# round_marker() in src/analysis/profiler.py was defined but never called
# from generate(). These helpers emit NVTX ranges around each phase so
# torch.profiler captures per-phase wall-time as user annotations in
# prof.events() (not key_averages, which only sees operator ops).
# No-ops on CPU / when CUDA NVTX is unavailable.
# ---------------------------------------------------------------------------

def _nvtx_push(label: str) -> None:
    """Push an NVTX range. Wrapped in try/except — older torch builds
    or non-CUDA devices skip silently."""
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_push(label)
        except (AttributeError, RuntimeError):
            pass


def _nvtx_pop() -> None:
    """Pop the most recent NVTX range. Safe to call even if push failed."""
    if torch.cuda.is_available():
        try:
            torch.cuda.nvtx.range_pop()
        except (AttributeError, RuntimeError):
            pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RASDConfig:
    """All knobs needed for one ablation run."""

    # Models
    target_model_name: str = "meta-llama/Llama-2-7b-hf"
    draft_model_name:  str = "princeton-nlp/Sheared-LLaMA-1.3B"
    # Optional HF revisions (commit hash or tag). None = HEAD at load time.
    # Pinning these makes runs reproducible against weight updates.
    target_revision: Optional[str] = None
    draft_revision:  Optional[str] = None

    # Speculative decoding
    spec_steps: int = 4                  # k — draft tokens per round (A2)
    temperature: float = 1.0
    top_p: float = 1.0

    # KV-cache ring communication
    kv_block_size: int = 512             # tokens per KV block (A3)
    prefetch_depth: int = 1              # 0=sync, 1=async-1, 2=async-2 (A4)

    # Generation
    max_new_tokens: int = 256

    # MLSys B3 — generate EXACTLY max_new_tokens, ignoring EOS.
    # ARM4 sets this so every cell produces the same number of rounds and
    # throughput ratios are comparable within the ladder. Acceptance is a
    # per-round quantity and is unaffected. Default False -> M3/M4 replay
    # behaviour (stop at EOS) is byte-identical.
    ignore_eos: bool = False
    dtype: str = "bfloat16"             # "float16" | "bfloat16"

    # Context length — used to decide RoPE scaling at model load.
    # Llama-2 native max_position_embeddings = 4096; anything larger needs
    # rope_scaling. 0 / None = no override (use the model's native max).
    context_length: int = 0

    # RoPE scaling strategy when context_length > native_max_position.
    # M3 used "linear" (factor = ceil(ctx/native)). Linear interpolation
    # is known to degrade quality past factor ≈ 16; for 1M context
    # (factor=256 over Llama-2's 4k), YaRN is recommended.
    #   "linear"   — linear position interpolation (M3 default; preserved)
    #   "yarn"     — YaRN (Peng et al. 2024), preferred for factor > 16
    #   "dynamic"  — NTK-aware dynamic scaling
    rope_type: str = "linear"

    # MLSys B1 — EXPLICIT extrapolation factor, independent of context length.
    #
    # The automatic path only scales when context_length exceeds the model's
    # native window, which cannot express "hold the context at 128k but vary
    # the RoPE scaling 1 -> 2 -> 4". That matched-context control is the one
    # that separates RoPE extrapolation from context length within a FIXED
    # model, so the factor is settable directly.
    #
    # Semantics:
    #   None  -> automatic (factor = ceil(ctx / native)); default, unchanged
    #   <=1.0 -> NO scaling applied; the model's own rope block is preserved.
    #            YaRN at factor 1 is the mathematical identity, and leaving
    #            Llama-3.1's native 'llama3' block in place is the only
    #            reading of "factor 1" that means "no extrapolation". This is
    #            what makes the ARM4 128k f1 cell a faithful Arm2 replication
    #            in RoPE terms.
    #   >1.0  -> YaRN over the model's native window with exactly this factor,
    #            regardless of context length.
    rope_factor: Optional[float] = None

    # MLSys — YaRN anchor override.
    #
    # `original_max_position_embeddings` in a YaRN dict is the base window the
    # scaling is computed AGAINST: the frequency bands are interpolated for a
    # length of factor x anchor. Two different anchors are defensible for
    # Llama-3.1-8B beyond its 131072 window:
    #
    #   None    -> anchor = the loaded model's max_position_embeddings
    #              (131072 for Llama-3.1-8B). "N x the window the model was
    #              trained for" — the ARM4 f2/f4 treatment.
    #   int     -> anchor = this value. In particular 8192, Llama-3.1's
    #              *pretraining* base: the length the frequencies were
    #              originally fit for and the base its own native `llama3`
    #              block names. Anchoring here keeps the base consistent with
    #              the model's own scaling rather than re-basing it.
    #
    # Both are candidates, not a settled choice: which one preserves the
    # target's coherence past 128k is an empirical question, so they are
    # gated by scripts/mlsys_coherence_gate.py before any speculative run.
    # Ignored for non-YaRN rope types. None preserves all earlier behaviour.
    rope_anchor_base: Optional[int] = None

    # Quantisation (to fit draft + target on same GPUs)
    quantize_draft: bool = True          # 4-bit NF4 via bitsandbytes
    quantize_target: bool = False

    # MLSys Phase 1 — draft context window cap.
    # Hard upper bound (in tokens) on the draft model's context window.
    # None (default) = use the draft's native max_position_embeddings, which
    # is the exact M3/M4 behaviour. When set, the effective window is
    # min(native, cap). Used to run the SAME draft model with a capped
    # window (arm 2) vs its full native window (arm 3), so the draft-window
    # effect can be measured in isolation from the target's RoPE regime.
    draft_window_cap: Optional[int] = None

    # MLSys Phase 3 — draft weight precision.
    #   "auto" (default) = follow `quantize_draft` (exact M3/M4 behaviour)
    #   "nf4"            = force 4-bit NF4 draft (requires CUDA)
    #   "bf16"           = force unquantized draft at cfg.torch_dtype
    # Used to test whether NF4 draft/target logit divergence — rather than
    # context length — drives the per-round cost increase.
    draft_dtype: str = "auto"

    # Reproducibility
    seed: int = 42

    # Debug
    debug: bool = False

    # M4 mentor-required sidecars (default off → M3 replay byte-identical)
    # When True, generate() returns metrics["per_token_trace"] with one
    # entry per spec round describing which draft tokens were accepted
    # and where they sat in the global sequence. Source for Figure 4
    # (acceptance rate vs token position).
    log_per_token: bool = False

    # MLSys analysis-plan — return the raw generated token IDs in the metrics
    # dict so the caller can persist them. Losslessness is a token-level claim
    # and decoded text is not injective, so the IDs must survive the run.
    # Default off: keeps M3/M4 replay byte-identical, and the metrics dict is
    # logged to wandb, which cannot take a non-scalar.
    save_generated_tokens: bool = False

    # M4 C6 — generation checkpoint/resume.
    # checkpoint_every == 0 disables (M3 byte-identical default).
    # When > 0, save verify-loop state every N rounds to
    # `<checkpoint_dir>/<run_id>/round_<n>.pt`. On generate() entry,
    # if a checkpoint exists for this run_id, restore state and skip
    # prefill — saves the bulk of pod time at 1M context.
    checkpoint_every: int = 0
    checkpoint_dir:   Optional[str] = None
    run_id:           Optional[str] = None

    # M4 C11 — NF4 KV-cache (default off; M3 byte-identical when off).
    # When True, the patched LlamaAttention forward routes K/V through
    # an NF4 quantize→dequantize round-trip on every step. This is the
    # "lossy bf16 path" — exercises the codec on real attention
    # activations and lets us measure α/PPL impact. Actual cache-storage
    # savings (subclassing DynamicCache or external NF4 store) are
    # M4 Phase C work; the codec behavior is validated here on every
    # cell of the matrix.
    kv_quant: bool = False
    # M4 Phase C 2026-05-10: outlier-keep mitigation for NF4 acceptance drop.
    # When kv_quant=True, the rank that holds global position 0 (rank 0
    # under sequence-parallel sharding) keeps the first
    # `kv_outlier_prefix_size` tokens in bf16 instead of NF4. These
    # "attention sinks" (StreamingLLM, Xiao et al. 2024) are dispropor-
    # tionately attended to throughout the model and are the largest
    # source of acceptance loss when quantized.
    # 0 = no outlier-keep (pure NF4 on every rank);
    # 128 = StreamingLLM default (8 MB / rank, negligible memory cost).
    kv_outlier_prefix_size: int = 128
    # Block size for the NF4 codec. Smaller block_size = more scales =
    # better per-block dynamic-range fit at the cost of slightly more
    # memory. block=32 lands at ~7% rel_err on real Llama K/V vs ~11%
    # for block=64; ~12% extra storage (0.625 vs 0.5625 bytes/elem).
    # The trade is worth it for acceptance — see Phase C blocker
    # 2026-05-10 NF4 acceptance drop.
    kv_block_size_nf4: int = 32
    # M4 Phase C 2026-05-10 NF4 chunked quantization. When > 0, the
    # NF4 cache's update() call splits the bf16 input along the
    # sequence axis into chunks of this size and quantizes each
    # in turn, freeing the slice between chunks. At 1M context this
    # drops the quant-path peak per layer from ~1.5 GB (S_local=128k
    # full) to ~24 MB per chunk — the freed headroom directly offsets
    # the FFN's ~9 GB transient that the trace identified as the
    # actual OOM contributor. Pass 0 for the legacy single-shot path
    # (M3 byte-identical). 2048 is the recommended value at long
    # context (matches kv_block_size for ring rotation granularity).
    nf4_update_chunk_size: int = 2048
    # Fold a new NF4 chunk into the previous one while both are shorter than
    # this many tokens (R1). The target-only decode appends one token per step,
    # so without it the chunk count -- and therefore the per-step dequantize
    # cost -- grows with the number of steps. The fold is exact. 0 disables.
    nf4_tail_merge_below: int = 64
    # Memory tracing for paper Figure 3 / attribution. When True,
    # MemoryTracer.snapshot() is called at lifecycle points in
    # generate() and a JSON sidecar is written to
    # <output_dir>/memory_trace/<run_id>.rank<rank>.json. Off by
    # default — M3 byte-identical when this is False.
    memory_trace: bool = False
    memory_trace_dir: Optional[str] = None

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16


# ---------------------------------------------------------------------------
# Token sampling helpers
# ---------------------------------------------------------------------------

def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample one token from logits (B, vocab). Returns (B,)."""
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = logits.scatter(-1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _acceptance_mask(
    draft_tokens: torch.Tensor,    # (B, k)  — draft model token IDs
    target_logits: torch.Tensor,   # (B, k+1, vocab_target)
    draft_logits: torch.Tensor,    # (B, k, vocab_draft)
    temperature: float,
) -> Tuple[torch.Tensor, int]:
    """Standard speculative decoding accept/reject criterion (Leviathan et al.).

    For each draft position i, accept with probability:
        min(1, p_target(x_i) / p_draft(x_i))

    All draft and target models share the LLaMA-2 SentencePiece tokenizer
    (vocab=32000), so token IDs are always directly comparable.

    Returns
        accepted : (B, k) bool tensor
        n_accepted: number of accepted tokens (first rejection position, batch=1)
    """
    B, k = draft_tokens.shape
    eps = 1e-9

    if temperature == 0.0:
        # Greedy: accept iff target argmax == draft token
        target_tokens = target_logits[:, :k].argmax(dim=-1)   # (B, k)
        accepted = (target_tokens == draft_tokens)
    else:
        target_probs = F.softmax(target_logits[:, :k] / temperature, dim=-1)
        draft_probs  = F.softmax(draft_logits / temperature, dim=-1)
        idx = draft_tokens.unsqueeze(-1)                       # (B,k,1)
        p_t = target_probs.gather(-1, idx).squeeze(-1)         # (B,k)
        p_d = draft_probs.gather(-1, idx).squeeze(-1)          # (B,k)
        accept_prob = torch.clamp(p_t / (p_d + eps), max=1.0)
        r = torch.rand_like(accept_prob)
        accepted = r < accept_prob

    first_reject = (accepted[0] == False).nonzero(as_tuple=False)
    n_accepted = first_reject[0].item() if len(first_reject) > 0 else k

    return accepted, int(n_accepted)


def _build_rope_scaling_dict(rope_type: str, factor: float,
                             native_max: int) -> Dict:
    """Return the rope_scaling dict to set on hf_cfg for the given strategy.

    Pure function so it's unit-testable without loading a model. Returns
    the canonical key set transformers ≥ 4.42 expects:

      * "linear"  → {"type": "linear",  "factor": F}
      * "yarn"    → {"type": "yarn",    "factor": F,
                     "original_max_position_embeddings": native_max}
      * "dynamic" → {"type": "dynamic", "factor": F}  (NTK-aware)

    Beta and mscale parameters for YaRN are left at the transformers
    defaults — overrideable by the user setting them on the hf_cfg
    after-the-fact if needed for paper-quality runs.

    ``native_max`` is the LOADED target's own ``max_position_embeddings``
    (4096 for Llama-2-7B, 131072 for Llama-3.1-8B), and the caller derives
    ``factor = ceil(context_length / native_max)`` from the same value. It
    is a parameter rather than a constant precisely so the MLSys Phase-A
    extrapolation ladder can push Llama-3.1-8B past its own 131072 window:
    256k -> factor 2, 512k -> factor 4, both anchored on base 131072.
    Hardcoded 4096 here would silently mis-scale every Llama-3 cell.

    COMPOSITION CHOICE (Llama-3.x). Llama-3.1-8B ships its own rope block,
    ``{'rope_type': 'llama3', 'factor': 8.0,
      'original_max_position_embeddings': 8192}`` — that block IS the
    model's native long-context mechanism, and it is what the native arms
    (rope_type="none") rely on. When YaRN is requested we REPLACE that
    block outright with the yarn dict anchored on the full 131072 native
    window rather than trying to compose YaRN on top of the llama3
    rescaling. Rationale: stacking two frequency-rescaling schemes has no
    defined semantics in transformers (rope_scaling holds a single dict),
    and anchoring on 131072 makes the ladder a clean, interpretable
    extrapolation from the model's own window — factor N means "N x the
    window the model was trained on". The consequence is that the 256k/512k
    cells are NOT native-llama3-plus-YaRN; they are pure YaRN over 131072.
    That is the intended treatment (RoPE extrapolation) and is why the
    Phase-A 128k cell uses rope_type="none" — it must stay byte-identical
    to Arm2 to serve as the in-distribution anchor of the ladder.

    Raises ValueError on an unknown rope_type.
    """
    rt = rope_type.lower()
    if rt == "linear":
        return {"type": "linear", "factor": float(factor)}
    if rt == "dynamic":
        return {"type": "dynamic", "factor": float(factor)}
    if rt == "yarn":
        return {
            "type": "yarn",
            "factor": float(factor),
            "original_max_position_embeddings": int(native_max),
        }
    raise ValueError(
        f"Unknown rope_type={rope_type!r}; "
        f"expected one of 'linear', 'yarn', 'dynamic'"
    )


def _resolve_rope_anchor(rope_anchor_base: Optional[int],
                         native_max: int) -> int:
    """Resolve the YaRN `original_max_position_embeddings` to use.

    Pure function so the anchor choice is unit-testable without booting a
    model. `rope_anchor_base=None` returns `native_max`, preserving every
    pre-existing call path byte-for-byte. An explicit value wins, but must be
    a positive int and must not exceed the loaded window: a YaRN anchor
    LARGER than the model's own window would ask for interpolation over a
    length the model cannot address, and would silently mis-scale rather
    than fail, so it is rejected.
    """
    if rope_anchor_base is None:
        return int(native_max)
    try:
        anchor = int(rope_anchor_base)
    except (TypeError, ValueError):
        raise ValueError(
            f"rope_anchor_base={rope_anchor_base!r} is not an integer"
        ) from None
    if anchor <= 0:
        raise ValueError(f"rope_anchor_base={anchor} must be positive")
    if anchor > native_max:
        raise ValueError(
            f"rope_anchor_base={anchor} exceeds the model's own window "
            f"({native_max}); a YaRN anchor beyond the trained window asks "
            f"for interpolation over an unaddressable length."
        )
    return anchor


def _rope_anchor_channel(rope_type: str) -> str:
    """Where transformers 4.47.1 actually reads a long-context anchor.

    Verified against the installed implementation, not the docs:

      * ``"llama3"`` -> ``"dict"``. ``_compute_llama3_parameters`` reads
        ``rope_scaling["original_max_position_embeddings"]`` and lists it as a
        REQUIRED key. Writing the anchor into the rope_scaling dict works.
      * ``"yarn"``   -> ``"config"``. ``_compute_yarn_parameters`` carries a
        ``TODO (joao): use the new `original_max_position_embeddings` from
        rope_scaling`` and never reads that key. Its interpolation band comes
        from ``config.max_position_embeddings``. Writing the anchor into the
        dict alone is therefore SILENTLY INERT for YaRN — the value has to go
        on the config.
      * anything else -> ``"unused"``: no anchor concept.

    This distinction is the difference between a rung that measures a rope
    configuration and one that measures nothing at all, so it is pinned here
    as a pure function and asserted by tests.
    """
    rt = (rope_type or "").lower()
    if rt == "llama3":
        return "dict"
    if rt == "yarn":
        return "config"
    return "unused"


def _top1_top2_gap(logits: torch.Tensor) -> list:
    """Top-1 minus top-2 logit at each position, as plain floats.

    Why this is recorded: under greedy decoding two runs that see the same prompt
    must choose the same token, but a target that is INDIFFERENT between the top
    two candidates -- a gap near zero -- can choose either for arithmetic reasons
    (reduction order, and the ring's non-associative online-softmax merge, which
    is why the engine broadcasts logits between ranks at all). Calling such a
    divergence a losslessness failure would report a numerics artefact as an
    implementation defect, so the gap is what decides tie versus mismatch.

    `logits` is (B, S, vocab) and the returned list has length S, one gap per
    position, in the order of the emitted tokens.
    """
    if logits is None or logits.numel() == 0:
        return []
    top2 = torch.topk(logits.float(), k=2, dim=-1).values
    return [float(g) for g in (top2[..., 0] - top2[..., 1])[0].tolist()]


def _step_gap(logit) -> list:
    """The gap at ONE position, from a `(B, vocab)` step logit.

    The target-only loop and the seed token both sample from a single position
    and both must record its gap, so they call the SAME function: two call sites
    computing the same quantity under different conventions is how two arms come
    to disagree about what a gap is, and the tie rule compares them.
    """
    if logit is None:
        return []
    return _top1_top2_gap(logit.unsqueeze(1))


def _round_emitted_gaps(target_logits_v, n_emit: int, n_acc: int,
                        with_bonus: bool) -> list:
    """Gaps for exactly the tokens one verify round EMITS, in emission order.

    A round emits `n_emit` accepted draft tokens and, when there was budget, one
    bonus token sampled from the target at position `n_acc`. So the list has
    exactly `n_emit + with_bonus` entries, which is what makes `token_gaps` line
    up element-for-element with the ids appended to `generated`. The tie rule
    reads the gap at the position where two arms first disagree, and an
    off-by-one there reports the wrong token's indifference.

    A round truncated by the budget emits no bonus and is not padded: its emitted
    tokens are the first `n_emit` positions, and the verified-but-unemitted tail
    is not part of the generation.
    """
    if target_logits_v is None:
        return []
    gaps = [float(g) for g in _top1_top2_gap(target_logits_v[:, :n_emit, :])]
    if with_bonus:
        gaps += [float(g) for g in
                 _top1_top2_gap(target_logits_v[:, n_acc:n_acc + 1, :])]
    return gaps


def _round_commit_plan(budget: int, n_acc: int, gamma: int) -> tuple:
    """How many tokens a verify round may commit, given the remaining budget.

    Returns `(n_emit, committed, with_bonus, truncated)`:

      n_emit      accepted draft tokens actually emitted
      committed   total tokens emitted this round (n_emit + the bonus token)
      with_bonus  whether there was room for the bonus/resampled token
      truncated   the round was cut short by the budget

    A round yields up to `n_acc + 1` tokens, so an uncapped final round
    overshoots `max_new_tokens` by up to gamma. That makes the generated length
    a function of how much the draft happened to be accepted, which breaks
    three things at once: the `prompt + BOS + generated == context` identity,
    the fixed-length contract the analysis plan pre-registers, and the
    losslessness comparison (the target-only arm stops exactly on the cap, so
    every speculative cell would look incomplete). Pure function so the
    arithmetic is testable without a GPU — the engine itself requires CUDA.

    `budget` is at least 1 whenever a round starts, because the loop condition
    is `generated < max_new_tokens`.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1 to start a round; got {budget}")
    if n_acc < 0 or n_acc > gamma:
        raise ValueError(f"n_acc {n_acc} outside [0, {gamma}]")
    n_emit = min(n_acc, budget)
    with_bonus = budget > n_acc
    committed = n_emit + (1 if with_bonus else 0)
    return n_emit, committed, with_bonus, committed < (n_acc + 1)


def _build_per_token_record(
    round_idx: int,
    global_pos_start: int,
    spec_steps: int,
    n_acc: int,
    draft_seq: torch.Tensor,
    accepted: torch.Tensor,
) -> Dict:
    """One row of the per-position acceptance sidecar (.jsonl entry).

    Pure function so it can be unit-tested without booting an engine.
    Schema is the contract Figure 4 (α vs token position) reads from.

    Args:
        round_idx          : 0-indexed verify round
        global_pos_start   : position of cur_token at round start (the
                             k draft tokens span global_pos_start+1..+k)
        spec_steps         : k — number of draft tokens proposed
        n_acc              : how many of the k were accepted (prefix)
        draft_seq          : (B, k) draft token IDs
        accepted           : (B, k) bool mask from _acceptance_mask

    `ended_on_eos` is always present and defaults to False; the verify loop
    flips it to True on the round whose bonus token was EOS and therefore
    terminated generation. It could not be computed here because the bonus
    token is sampled AFTER this record is appended (and appended before the
    EOS test so that a mid-round checkpoint captures the current round). The
    key is emitted unconditionally so every record has the same schema and
    post-hoc analysis never has to distinguish "absent" from "False".
    """
    return {
        "round_idx":        int(round_idx),
        "global_pos_start": int(global_pos_start),
        "spec_steps":       int(spec_steps),
        "n_acc":            int(n_acc),
        "draft_tokens":     draft_seq[0].tolist(),
        "accepted":         [bool(x) for x in accepted[0].tolist()],
        "ended_on_eos":     False,
    }


# ---------------------------------------------------------------------------
# Numerics provenance: what the RUNNING objects are, not what the config asked
# for
# ---------------------------------------------------------------------------

# The canonical spelling of every dtype the campaign can produce. Recorded in a
# row so a reader can compare two engines' precision, which only works if the
# spellings are the same: `bf16` and `bfloat16` are the same dtype and must
# compare equal.
_DTYPE_ALIASES = {
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp16": "float16", "float16": "float16", "half": "float16",
    "fp32": "float32", "float32": "float32", "float": "float32",
    "fp64": "float64", "float64": "float64", "double": "float64",
    "int8": "int8", "uint8": "uint8",
}


def normalize_dtype_name(dtype) -> str:
    """One canonical spelling per dtype, so equal dtypes compare equal."""
    if dtype is None:
        return ""
    name = getattr(dtype, "__name__", None) or str(dtype)
    name = name.rsplit(".", 1)[-1].strip().lower()
    return _DTYPE_ALIASES.get(name, name)


def detect_weight_precision(model) -> str:
    """How the weights are ACTUALLY stored, read off the loaded model.

    Why not the config: `quantize_target=True` is an instruction, and this
    project has already shipped a run whose config said one thing while the
    loaded weights were another (the MPS/CPU path warns and silently skips
    4-bit). A row that reports its own config cannot detect that, and the whole
    point of the column is to let a reader see the precision difference that
    makes two engines' outputs incomparable.

    Order matters. A 4-bit model's parameters are `Params4bit` (uint8 storage
    behind a dequantize hook), so reading `next(model.parameters()).dtype`
    first would report `uint8` -- or, worse, the compute dtype -- and lose the
    distinction between fp4 and nf4 storage, which is exactly the axis the
    vLLM comparison is about.

    Returns "fp4"/"nf4"/"int8"/"bfloat16"/... ; "" when there is no model.
    """
    if model is None:
        return ""
    if getattr(model, "is_loaded_in_4bit", False):
        qc = getattr(getattr(model, "config", None), "quantization_config", None)
        # Attribute FIRST, then mapping. A real BitsAndBytesConfig is a plain
        # object (`QuantizationConfigMixin`), while a config round-tripped
        # through a saved `config.json` arrives as a dict -- so both spellings
        # occur and this has to handle either. Checking `isinstance(qc, dict)`
        # first would read a dict that also carries the value as an attribute
        # (which is how a `dict` subclass behaves) as "unset" and report the
        # nf4 default for an fp4 model: exactly the mislabelling this column
        # exists to prevent.
        qtype = getattr(qc, "bnb_4bit_quant_type", None)
        if qtype is None and isinstance(qc, dict):
            qtype = qc.get("bnb_4bit_quant_type")
        # bitsandbytes defaults to nf4 when the type is unset; iff it is 4-bit
        # loaded and says nothing, the storage is nf4.
        return str(qtype or "nf4").lower()
    if getattr(model, "is_loaded_in_8bit", False):
        return "int8"
    try:
        param = next(model.parameters())
    except (StopIteration, AttributeError):
        return ""
    return normalize_dtype_name(param.dtype)


def detect_kv_precision(cache) -> str:
    """The KV-cache precision in use, read off the CACHE OBJECT.

    The class is the authority, not the flag that built it: `kv_quant=True` with
    an empty cache falls back to HF's bf16 `DynamicCache` inside the model, and a
    legacy tuple is bf16 by construction (it is built from dequantized tensors).
    So the name comes from what is actually holding the cache.

    Returns "nf4" | "bfloat16" | ... | "" when there is no cache.
    """
    if cache is None:
        return ""
    # Import here: this module is imported by CPU-only tests, and the cache
    # module pulls in the codec it needs, not torch CUDA.
    from src.models.nf4_dynamic_cache import NF4DynamicCache
    if isinstance(cache, NF4DynamicCache):
        return "nf4"
    layer0 = None
    try:
        layer0 = cache[0]
    except Exception:                                      # noqa: BLE001
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            # A cache that reports a length but exposes no layer view is one we
            # cannot name; report it as unknown rather than guessing.
            return "unknown"
        return ""
    try:
        k = layer0[0] if isinstance(layer0, (tuple, list)) else layer0
        return normalize_dtype_name(k.dtype)
    except Exception:                                      # noqa: BLE001
        return ""


def numerics_report(model, cache) -> Dict[str, str]:
    """The precision columns for one run: weights and KV, both measured."""
    return {
        "weight_precision": detect_weight_precision(model),
        "kv_dtype": detect_kv_precision(cache),
    }


def _draft_window(input_ids, window: int):
    """The sequence the DRAFT is conditioned on: leading BOS + most recent tokens.

    Why the leading token is kept rather than simply taking the last `window`
    tokens: the engine builds its input with `tokenizer(prompt)`, which prepends
    the BOS, and position 0 is the attention sink the whole model is trained
    around. Dropping it makes the draft's first position a mid-document token
    whose rotary phase is 0 but whose identity is wrong, which is a different
    (and worse) conditioning than the target ever sees.

    Why this exists at all: `generate_text` used to build its own draft input
    with `tokenizer(prompt, max_length=draft_max_len, truncation=True)`, and HF
    truncation with the default `truncation_side='right'` keeps the FIRST
    `draft_max_len` tokens. At any context longer than the draft window the
    draft was therefore conditioned on the OPENING of the document while the
    target was conditioned on its ending -- and the manuscript describes the
    opposite ("attends over only the most recent 4,096 tokens"). Making
    `generate` the only place the window is applied removes the possibility of
    the two disagreeing.

    `window` is the draft's effective window (`draft_max_len`, already capped by
    `draft_window_cap`). A prompt that already fits is returned unchanged, so
    short-context runs are bit-identical to before.
    """
    if window <= 0 or input_ids.shape[1] <= window:
        return input_ids
    keep = window - 1
    if keep <= 0:
        return input_ids[:, :1]
    return torch.cat([input_ids[:, :1], input_ids[:, -keep:]], dim=1)


def _broadcast_round_state(draft_seq, target_logits_v, draft_logits,
                           world_size: int) -> None:
    """Fix one round's draft tokens and logits to rank 0's, on every rank.

    Ring attention's online-softmax merges K/V slices in a rank-DIFFERENT order,
    and floating-point merges are not associative, so `target_logits_v` drifts
    between ranks. When `accept_prob = p_target/p_draft` lands near a `r ~ U[0,1)`
    threshold, ranks would then flip independently: `n_acc` differs, so
    `_truncate_kv` leaves different local cache sizes and the next round's ring
    P2P sizes mismatch. That is the coalesced-timeout path (SeqNum ~3500-3600)
    this broadcast exists to close.

    The DRAFT TOKENS belong in the same set. Every rank runs the draft model, so
    its tokens agree under greedy decoding -- which is why leaving `draft_seq`
    out went unnoticed -- but at temperature > 0 each rank samples its own, and
    the accept/reject then tests different proposals per rank. Sharing the
    logits is not enough to make `n_acc` agree if the proposals differ.

    One helper rather than three inline calls so the SET is testable: a missing
    member is invisible in the round's output and only shows up as a hang at
    round 3500.
    """
    if world_size <= 1:
        return
    dist.broadcast(draft_seq, src=0)
    dist.broadcast(target_logits_v, src=0)
    dist.broadcast(draft_logits, src=0)


def _kv_seq_len(past_kv) -> int:
    """Positions a cache currently holds, read back FROM the cache.

    The committed length is a claim about the cache, so it is measured rather
    than recomputed from the arithmetic that was supposed to produce it: an
    off-by-one in `_truncate_kv` (a dropped bonus, an un-emitted verified tail,
    a truncation that did not apply to the NF4 store) would otherwise be
    invisible, because the same expression would appear on both sides of the
    comparison.

    NF4DynamicCache and HF's DynamicCache both expose `get_seq_length()`; a
    legacy tuple can only be measured from its own tensor.
    """
    if past_kv is None:
        return 0
    getter = getattr(past_kv, "get_seq_length", None)
    if callable(getter):
        try:
            return int(getter())
        except Exception:                                  # noqa: BLE001
            pass
    try:
        return int(past_kv[0][0].shape[2])
    except Exception:                                      # noqa: BLE001
        return 0


def _truncate_kv(past_kv, new_len: int):
    """Truncate past_key_values to `new_len` positions along the seq dim.

    Three cases:
      * NF4DynamicCache (M4 C11) — call its in-place `truncate(new_len)`
        method and return the same instance. Crucial for preserving
        NF4 storage across rounds; otherwise the next forward sees a
        bf16 legacy tuple and we lose the ~3.55x memory savings.
      * HF DynamicCache (and other types with an iterable layer view) —
        produce a new legacy tuple of bf16 tensors. Backward-compatible
        with the M3 code path.
      * Legacy tuple — same as above.
    """
    if past_kv is None:
        return None
    # NF4 cache: in-place truncation preserves the storage type
    if hasattr(past_kv, "truncate") and callable(getattr(past_kv, "truncate")):
        past_kv.truncate(new_len)
        return past_kv
    # Legacy tuple or HF DynamicCache: build a new legacy tuple
    out = []
    for layer in past_kv:
        k, v = layer[0], layer[1]
        out.append((k[:, :, :new_len, :].contiguous(),
                    v[:, :, :new_len, :].contiguous()))
    return tuple(out)


# ---------------------------------------------------------------------------
# Main RASD class
# ---------------------------------------------------------------------------

class RASDInference:
    """Ring Attention Speculative Decoding inference engine.

    Loads target and draft models, owns the three CUDA streams, and exposes
    a `generate()` method that implements the RASD decoding loop.

    Metrics collected per generate() call (returned as dict):
        tokens_generated   total new tokens produced
        time_sec           wall time for generation
        throughput_tps     tokens per second
        acceptance_rate    mean fraction of draft tokens accepted
        mean_latency_ms    mean per-token latency
        gpu_peak_mem_mb    peak GPU memory during generation
    """

    def __init__(self, config: RASDConfig):
        self.cfg = config
        torch.manual_seed(config.seed)

        self._setup_streams()
        self._load_models()

        # Ring state (set when distributed is active)
        self._rank       = dist.get_rank()       if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        # Optional dedicated NCCL sub-group (legacy carry-over; unused after
        # ring moved into the attention forward). Kept None for compatibility.
        self._signal_group = None

        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug("[RASD] init complete — debug mode ON (forced sync after each stream op)")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_streams(self):
        """Create the two CUDA streams used by the verify loop.

        After R3 (ring lives inside LlamaAttention.forward), there is no
        separate KV-ring communication stream — P2P rotation runs as part
        of the target verify forward on stream_compute. Only two streams
        survive: target compute and draft generation.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("RASD requires CUDA.")
        self.stream_compute = torch.cuda.Stream()   # target model forward
        self.stream_draft   = torch.cuda.Stream()   # draft model forward

    def _build_hf_config(self, model_name: str, revision: Optional[str],
                         context_length: int, label: str,
                         apply_rope_scaling: bool = True,
                         rope_type: str = "linear",
                         rope_factor: Optional[float] = None,
                         rope_anchor_base: Optional[int] = None):
        """Load the model's HF config and apply RoPE scaling if needed.

        Llama-2 ships with max_position_embeddings=4096. To run at longer
        contexts (e.g. 64k for the M3 ablation, 1M for M4), we set
        rope_scaling with factor = ceil(ctx / native_max).

        `rope_type` selects the scaling strategy (default linear, preserved
        from M3):
          * "linear"   — Position interpolation. Works well up to
                         factor ≈ 16 (e.g. 64k from 4k). Degrades past
                         that.
          * "yarn"     — Peng et al. 2024. Frequency-aware scaling that
                         maintains quality at large factors. Recommended
                         for 1M context (factor=256 over Llama-2-7B's 4k).
          * "dynamic"  — NTK-aware dynamic scaling (uses the original
                         theta with rescaled frequencies).
          * "none"     — apply NO scaling; raise if ctx exceeds the native
                         window. Used by the MLSys native-context arms.

        `apply_rope_scaling` defaults to True (target). For the **draft**
        model we pass False: capping the draft at its native context cap
        (Sheared-LLaMA-1.3B = 4096) saves ~11 GB/rank of replicated KV at
        ctx=64k vs scaling the draft to match the target. Speculative
        decoding tolerates a smaller draft window — `_draft_window` in
        generate() already gives the draft the leading BOS plus the most
        recent `draft_max_len - 1` tokens of context. R6.4 OOM analysis (2026-05-06) showed draft KV
        was eating ~12 GB/rank at ctx=64k under the old behaviour.
        """
        from transformers import AutoConfig
        import math

        hf_cfg = AutoConfig.from_pretrained(model_name, revision=revision)
        # MLSys Phase 1 — explicit "no scaling" mode. The native arms
        # (Llama-3.1-8B @128k) must run inside the model's own window with
        # NO rope_scaling, otherwise the experiment silently stops
        # separating "native context" from "YaRN/OOD". Fail loud if the
        # requested context does not actually fit the native window.
        if rope_type == "none" and apply_rope_scaling:
            if context_length and context_length > hf_cfg.max_position_embeddings:
                raise ValueError(
                    f"rope_type='none' but context_length={context_length} exceeds "
                    f"{label} native max_position_embeddings="
                    f"{hf_cfg.max_position_embeddings}. Refusing to silently "
                    f"apply scaling — fix the config or use 'yarn'/'linear'."
                )
            logger.info(
                "[RoPE] %s: rope_type='none' — ctx=%d within native_max=%d, "
                "no scaling applied.",
                label, context_length, hf_cfg.max_position_embeddings,
            )
            return hf_cfg
        # MLSys B1 — explicit factor, independent of context length. Handled
        # BEFORE the automatic branch so a fixed-context ladder (128k at
        # factor 1/2/4) is expressible.
        if apply_rope_scaling and rope_factor is not None:
            native_max = hf_cfg.max_position_embeddings
            f = float(rope_factor)
            if f <= 1.0:
                # Factor 1 == identity == no extrapolation. Preserve the
                # model's OWN rope block (Llama-3.1's 'llama3') rather than
                # replacing it with an identity YaRN dict, so this cell is a
                # faithful in-distribution anchor.
                logger.info(
                    "[RoPE] %s: rope_factor=%.3g <= 1 — native rope block "
                    "preserved (no scaling), native_max=%d",
                    label, f, native_max,
                )
                return hf_cfg
            if rope_type == "llama3":
                # Robustness rung: stay INSIDE Meta's own mechanism and vary
                # only the factor, so a change here cannot be attributed to
                # swapping the rope implementation. Copy the model's shipped
                # llama3 dict and override only `factor`.
                base = getattr(hf_cfg, "rope_scaling", None)
                if not isinstance(base, dict) or base.get("rope_type") != "llama3":
                    raise ValueError(
                        f"rope_type='llama3' requested for {model_name} but its "
                        f"shipped rope_scaling is {base!r}; this rung requires a "
                        f"model that ships a llama3 rope block."
                    )
                d = dict(base)
                d["factor"] = f
                if rope_anchor_base is not None:
                    # llama3 reads the anchor from the dict, so this is the
                    # effective channel for this rope type.
                    d["original_max_position_embeddings"] = _resolve_rope_anchor(
                        rope_anchor_base, native_max,
                    )
                hf_cfg.rope_scaling = d
            else:
                eff_rt = rope_type if rope_type not in ("none",) else "yarn"
                anchor = _resolve_rope_anchor(rope_anchor_base, native_max)
                hf_cfg.rope_scaling = _build_rope_scaling_dict(
                    eff_rt, f, anchor,
                )
                # Put the anchor where this rope type actually reads it. For
                # YaRN the dict key is inert in transformers 4.47.1, so
                # without this the "true 8192 base" rung would be a silent
                # no-op identical to the re-based one.
                chan = _rope_anchor_channel(eff_rt)
                if chan == "config" and rope_anchor_base is not None:
                    hf_cfg.max_position_embeddings = anchor
                logger.info(
                    "[RoPE] %s: rope_type=%s anchored on base=%d via %s "
                    "(model native window=%d)",
                    label, eff_rt, anchor, chan, native_max,
                )
            # target_length may stay <= native_max; the window is what the
            # scaled frequencies are valid for, so raise it to at least the
            # scaled span for bookkeeping — EXCEPT when the anchor is being
            # carried on this field for YaRN, where overwriting it would
            # silently undo the anchor before the model ever reads it.
            anchored_via_config = (
                rope_anchor_base is not None
                and _rope_anchor_channel(rope_type) == "config"
            )
            if (context_length and context_length > native_max
                    and not anchored_via_config):
                hf_cfg.max_position_embeddings = context_length
            logger.info(
                "[RoPE] %s: EXPLICIT rope_factor=%.3g over native_max=%d "
                "-> %s (ctx=%d)",
                label, f, native_max, hf_cfg.rope_scaling, context_length,
            )
            return hf_cfg
        if (apply_rope_scaling and context_length
                and context_length > hf_cfg.max_position_embeddings):
            # native_max is the LOADED model's own window, so the ladder
            # generalises to any target: Llama-2-7B (4096) at 128k -> factor
            # 32, Llama-3.1-8B (131072) at 256k/512k -> factor 2/4.
            # NOTE: this REPLACES any rope block the model ships (Llama-3.x
            # has its own 'llama3' block) — see the composition note in
            # _build_rope_scaling_dict.
            native_max = hf_cfg.max_position_embeddings
            factor = float(math.ceil(context_length / native_max))
            anchor = _resolve_rope_anchor(rope_anchor_base, native_max)
            hf_cfg.rope_scaling = _build_rope_scaling_dict(
                rope_type, factor, anchor,
            )
            # Existing behaviour for every un-anchored call: advertise the
            # scaled span. An explicit YaRN anchor instead occupies this
            # field (it is the channel transformers reads), so it must not be
            # overwritten.
            if (rope_anchor_base is not None
                    and _rope_anchor_channel(rope_type) == "config"):
                hf_cfg.max_position_embeddings = anchor
            else:
                hf_cfg.max_position_embeddings = context_length
            logger.info(
                "[RoPE] %s: ctx=%d > native=%d → %s scaling factor=%.1f",
                label, context_length, native_max, rope_type, factor,
            )
        elif (not apply_rope_scaling and context_length
                and context_length > hf_cfg.max_position_embeddings):
            logger.info(
                "[RoPE] %s: skipping scaling (caller-disabled); native_max=%d "
                "stays — caller is expected to truncate inputs to that window.",
                label, hf_cfg.max_position_embeddings,
            )
        return hf_cfg

    def _load_models(self):
        """Load target and (optionally quantised) draft models.

        Handles three backends automatically via DeviceCapabilities:
          CUDA (RunPod) — 4-bit NF4 quantization, device_map per local_rank
          MPS (MacBook) — no quantization (bitsandbytes unsupported), .to("mps")
          CPU           — no quantization, .to("cpu")
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from src.utils.device import DeviceCapabilities

        cfg = self.cfg
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._caps = DeviceCapabilities.detect(local_rank=local_rank)
        self._device = self._caps.device

        target_hf_config = self._build_hf_config(
            cfg.target_model_name, cfg.target_revision, cfg.context_length,
            label="target", rope_type=cfg.rope_type,
            rope_factor=cfg.rope_factor,
            rope_anchor_base=getattr(cfg, "rope_anchor_base", None),
        )

        logger.info("Loading target model: %s  [device=%s]", cfg.target_model_name, self._device)
        # MLSys A5 — weight precision is pinned EXPLICITLY to "fp4".
        # bitsandbytes already defaults to fp4 when bnb_4bit_quant_type is
        # unset, so this changes no behaviour; it stops the default from
        # drifting silently, which matters because the papers describe these
        # weights as "NF4" (they are FP4). Do NOT switch this to "nf4": the
        # Arm1/2/3 results were produced under fp4 weights and changing it
        # would make the new arms incomparable.
        # NOTE the two distinct 4-bit mechanisms in this codebase:
        #   * WEIGHTS  -> bitsandbytes 4-bit FP4 (this config)
        #   * KV CACHE -> NF4 via the custom chunked codec (kv_quant=True)
        target_bnb = None
        if cfg.quantize_target and self._caps.supports_quantization:
            target_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=cfg.torch_dtype,
                bnb_4bit_quant_type="fp4",
            )
        elif cfg.quantize_target:
            logger.warning("quantize_target=True ignored — 4-bit requires CUDA (current: %s)",
                           self._caps.device_type)

        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_name,
            config=target_hf_config,
            revision=cfg.target_revision,
            torch_dtype=cfg.torch_dtype,
            quantization_config=target_bnb,
            **self._caps.hf_device_map_kwargs(),
        )
        if self._caps.device_type in ("mps", "cpu") and target_bnb is None:
            self.target_model = self.target_model.to(self._device)
        self.target_model.eval()

        # Install ring-attention forward on every LlamaAttention layer.
        # No-op when world_size <= 1 (preserves single-rank exactness).
        # A3 (cfg.kv_block_size) → per-step transmission chunk size.
        # A4 (cfg.prefetch_depth) → ring-step prefetch depth.
        from src.models.ring_llama_attention import install_ring_attention
        ws = dist.get_world_size() if dist.is_initialized() else 1
        rk = dist.get_rank()       if dist.is_initialized() else 0
        install_ring_attention(
            self.target_model,
            world_size=ws,
            rank=rk,
            chunk_size=cfg.kv_block_size,
            prefetch_depth=cfg.prefetch_depth,
            kv_quant=cfg.kv_quant,
        )

        # M4 Phase C 2026-05-10: target-only baseline mode. When
        # cfg.spec_steps == 0, we skip loading the draft model entirely
        # and generate() runs autoregressive single-token decode through
        # the target — same prefill, same NF4 cache, same ring attention,
        # but no speculation. This is the apples-to-apples baseline that
        # isolates the contribution of speculative decoding while keeping
        # every other variable (model, sequence parallelism, KV cache
        # format, outlier-keep) identical to RASD.
        self.draft_model = None
        self.draft_tokenizer = None
        self.draft_max_len = 0

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name, revision=cfg.target_revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if cfg.spec_steps == 0:
            logger.info(
                "[target-only] cfg.spec_steps=0 — skipping draft model load. "
                "Forward pass will run autoregressive decode via target only."
            )
            return

        # Don't RoPE-scale the draft: keeping it at its native context cap
        # (Sheared-LLaMA-1.3B = 4096) saves ~11 GB/rank of replicated draft
        # KV at ctx=64k. The `self.draft_max_len` truncation below already
        # feeds the draft only its native_max recent tokens. See
        # _build_hf_config docstring for the R6.4 OOM analysis that drove
        # this decision.
        draft_hf_config = self._build_hf_config(
            cfg.draft_model_name, cfg.draft_revision, cfg.context_length,
            label="draft", apply_rope_scaling=False,
        )

        logger.info("Loading draft model: %s  [device=%s]", cfg.draft_model_name, self._device)
        # MLSys Phase 3: an explicit draft_dtype wins over `quantize_draft`.
        # "auto" resolves to `quantize_draft`, so M3/M4 runs are unchanged.
        if cfg.draft_dtype not in ("auto", "nf4", "bf16"):
            raise ValueError(
                f"draft_dtype={cfg.draft_dtype!r} invalid; "
                f"expected one of 'auto', 'nf4', 'bf16'"
            )
        want_nf4_draft = (
            cfg.quantize_draft if cfg.draft_dtype == "auto"
            else cfg.draft_dtype == "nf4"
        )
        if want_nf4_draft != cfg.quantize_draft:
            logger.info(
                "[draft-dtype] %s override: quantize_draft=%s → %s",
                cfg.draft_dtype, cfg.quantize_draft, want_nf4_draft,
            )
        draft_bnb = None
        if want_nf4_draft and self._caps.supports_quantization:
            # Same explicit FP4 pin as the target (see the note above).
            draft_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=cfg.torch_dtype,
                bnb_4bit_quant_type="fp4",
            )
        elif want_nf4_draft:
            logger.warning("draft 4-bit requested (draft_dtype=%s) but ignored — "
                           "4-bit requires CUDA (current: %s)",
                           cfg.draft_dtype, self._caps.device_type)

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            cfg.draft_model_name,
            config=draft_hf_config,
            revision=cfg.draft_revision,
            torch_dtype=cfg.torch_dtype,
            quantization_config=draft_bnb,
            **self._caps.hf_device_map_kwargs(),
        )
        if self._caps.device_type in ("mps", "cpu") and draft_bnb is None:
            self.draft_model = self.draft_model.to(self._device)
        self.draft_model.eval()

        self.draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_name, revision=cfg.draft_revision)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token

        # Max sequence length the draft model supports
        # TinyLlama=2048, Sheared-LLaMA=4096 — all share a LLaMA-2 tokenizer
        self.draft_max_len = getattr(self.draft_model.config, "max_position_embeddings",
                             getattr(self.draft_model.config, "n_positions", 4096))

        # MLSys Phase 1 — optional hard cap on the draft context window.
        # This is the single place the draft window is SET, and `_draft_window`
        # in generate() is the single place it is APPLIED. generate_text() used
        # to apply a second, differently-directed truncation of its own, which
        # is why the draft's context did not match this number.
        if cfg.draft_window_cap is not None:
            cap = int(cfg.draft_window_cap)
            if cap <= 0:
                raise ValueError(f"draft_window_cap must be > 0, got {cap}")
            native = self.draft_max_len
            self.draft_max_len = min(native, cap)
            logger.info(
                "[draft-window] cap=%d tokens, native=%d → effective=%d",
                cap, native, self.draft_max_len,
            )

    # ------------------------------------------------------------------
    # M4 C6 — checkpoint helpers (no-ops when checkpoint_every == 0)
    # ------------------------------------------------------------------

    def _try_load_checkpoint(self):
        """Return the latest GenerationCheckpoint for this run+rank, or None.

        Returns None when:
          * cfg.checkpoint_dir is unset
          * cfg.run_id is unset
          * no checkpoint file exists for this (run_id, rank)
        """
        cfg = self.cfg
        if not cfg.checkpoint_dir or not cfg.run_id:
            return None
        from src.models.checkpoint import GenerationCheckpoint, latest_checkpoint
        path = latest_checkpoint(cfg.checkpoint_dir, cfg.run_id, rank=self._rank)
        if path is None:
            return None
        ckpt = GenerationCheckpoint.load(path)
        # Move tensors to the engine's device so the verify loop runs natively
        if hasattr(self, "_device"):
            ckpt = ckpt.move_tensors_to(self._device)
        return ckpt

    def _maybe_save_checkpoint(self, *, n_rounds, global_seqlen,
                               total_accepted, total_draft_toks,
                               cur_token, generated, past_kv, draft_past_kv,
                               per_token_trace, prefill_len):
        """Save a GenerationCheckpoint for the current rank, if scheduled.

        No-op when cfg.checkpoint_every == 0 (the default; preserves M3
        byte-identical behavior). All ranks save their own KV slice into
        per-rank files — the on-disk layout is rank-aware.

        NOTE a saved checkpoint can no longer be RESUMED: `generate` refuses a
        run it finds a checkpoint for, because the per-token logit gaps are not
        part of a checkpoint and a resumed run would write a misaligned gap
        array. The file remains readable by the checkpoint tooling; it is the
        in-generation resume that is refused.
        """
        cfg = self.cfg
        if cfg.checkpoint_every <= 0 or not cfg.checkpoint_dir or not cfg.run_id:
            return
        from src.models.checkpoint import (
            GenerationCheckpoint, checkpoint_path, should_save_this_round,
        )
        if not should_save_this_round(cfg.checkpoint_every, n_rounds):
            return
        # CRITICAL: when past_kv is an NF4DynamicCache, serialize it
        # NF4-native via to_serializable(). Iterating it the legacy way
        # (`for layer in past_kv`) calls __iter__ → _dequantize_layer,
        # which materializes the FULL bf16 cache on the GPU before
        # .cpu() — at ctx=1M × W=8 that's ~64 GB / rank, OOMs. Plus the
        # checkpoint file balloons to bf16 size (~3.55x larger than NF4
        # storage), and the resumed legacy tuple loses NF4 storage
        # permanently. (Fix for high-risk finding #1, 2026-05-10
        # third-pass review.)
        if hasattr(past_kv, "to_serializable"):
            serialized_past_kv = past_kv.to_serializable()
        else:
            serialized_past_kv = tuple(
                tuple(t.detach().cpu() for t in layer) for layer in past_kv
            )
        if draft_past_kv is None:
            # Target-only mode (spec_steps=0) doesn't load a draft model,
            # so draft_past_kv is None. Don't try to iterate it — that
            # was the 'NoneType is not iterable' bug from p35b ctx512k+
            # cells in Phase C; the bug only triggered when
            # checkpoint_every was set (e.g., ctx512k YAML override),
            # masking it from single-cell debug runs. Fix: serialize as
            # None and let _resume_from_checkpoint pass it through.
            serialized_draft_past_kv = None
        elif hasattr(draft_past_kv, "to_serializable"):
            serialized_draft_past_kv = draft_past_kv.to_serializable()
        else:
            serialized_draft_past_kv = tuple(
                tuple(t.detach().cpu() for t in layer) for layer in draft_past_kv
            )

        ckpt = GenerationCheckpoint(
            n_rounds=n_rounds,
            global_seqlen=global_seqlen,
            total_accepted=total_accepted,
            total_draft_toks=total_draft_toks,
            prefill_len=prefill_len,
            cur_token=cur_token.detach().cpu(),
            generated=[t.detach().cpu() for t in generated],
            past_kv=serialized_past_kv,
            draft_past_kv=serialized_draft_past_kv,
            per_token_trace=list(per_token_trace),
            # Save BOTH CPU and CUDA RNG state. Under temperature > 0
            # (the M3 default), _sample (torch.multinomial on CUDA
            # tensors) and _acceptance_mask (torch.rand_like on CUDA
            # accept_prob) consume CUDA RNG, not CPU. Saving only CPU
            # state would still let resumed runs diverge — the CUDA
            # generator advances independently. (Fix for finding #3
            # from 2026-05-10 review; the original Fix #5 only
            # addressed CPU.)
            rng_state=torch.get_rng_state(),
            cuda_rng_state=(
                torch.cuda.get_rng_state(self._device)
                if torch.cuda.is_available()
                and getattr(self, "_device", None) is not None
                and self._device.type == "cuda"
                else None
            ),
        )
        path = checkpoint_path(
            cfg.checkpoint_dir, cfg.run_id, round_idx=n_rounds, rank=self._rank,
        )
        ckpt.save(path)
        if self._rank == 0:
            logger.info("[checkpoint] saved %s", path)

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,                        # (B, S_prompt) — target tokenizer
        attention_mask: Optional[torch.Tensor] = None,
        draft_input_ids: Optional[torch.Tensor] = None, # (B, S_prompt) — draft tokenizer
    ) -> Tuple[torch.Tensor, Dict]:
        """RASD speculative decoding generation.

        Returns
            generated_ids : (B, S_prompt + max_new_tokens)
            metrics       : dict with throughput, acceptance_rate, etc.
        """
        cfg    = self.cfg
        device = input_ids.device
        B, S   = input_ids.shape

        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.perf_counter()

        print(f"[TRACE rank={self._rank}] generate() start, S={S}, block_size={cfg.kv_block_size}", flush=True)

        # Memory tracer (paper Figure 3 attribution). Off by default;
        # gated on cfg.memory_trace. Snapshot at every meaningful
        # lifecycle transition so consecutive deltas attribute peak
        # memory to its components (post-load, post-prefill, post each
        # verify round, end). The tracer is per-rank; only rank 0's
        # JSON is needed for the paper figure but all ranks emit so
        # asymmetries (e.g. rank-0 outlier-keep prefix) are visible.
        mem_tracer = None
        if cfg.memory_trace:
            from src.models.memory_trace import MemoryTracer
            trace_dir = cfg.memory_trace_dir or "results/memory_trace"
            mem_tracer = MemoryTracer(
                rank=self._rank,
                run_id=(cfg.run_id or f"rank{self._rank}_S{S}"),
                out_dir=trace_dir,
                device=device,
            )
            mem_tracer.snapshot("post_load",  S=S, S_local=S // max(self._world_size, 1))

        # ---- C6 RESUME: REFUSED ------------------------------------------
        # Generation always starts fresh. The per-token logit gaps are
        # accumulated in memory as tokens are emitted and are NOT part of a
        # checkpoint, so a resumed run would write a gap array covering only the
        # post-resume tokens beside a full-length id list: the alignment check
        # would abort it after the money was spent, and if the two lengths
        # happened to agree it would report the neighbouring token's
        # indifference at every divergence -- excusing real mismatches as
        # numerics ties. A found checkpoint is therefore REMOVED from the
        # decision and reported, never resumed.
        #
        # `checkpoint_every=0` for every campaign stage (asserted in the dry run
        # and in tests/test_mlsys_resume_refusal.py), so this is a guard against
        # a misconfiguration, not a path the campaign takes.
        if cfg.checkpoint_every > 0:
            existing = self._try_load_checkpoint()
            if existing is not None:
                raise RuntimeError(
                    f"resume refused: checkpoint {self.cfg.run_id!r} found in "
                    f"{self.cfg.checkpoint_dir!r}, but the per-token logit gaps "
                    f"cannot be restored from a checkpoint, so a resumed run "
                    f"would write a misaligned gap array. Re-run from scratch "
                    f"(checkpoint_every=0)."
                )

        # ============================================================
        # Fresh start — run target + draft prefill, sample seed token
        # ============================================================
        # Under multi-rank (R3 dual-cache layout), each rank owns the contiguous
        # slice [rank*S/W, (rank+1)*S/W) of the prompt. Each rank embeds and
        # forwards its own slice; the patched LlamaAttention performs ring
        # attention across ranks for cross-slice attention. Position IDs are
        # the absolute global positions so RoPE produces correct embeddings.
        if self._world_size > 1:
            # Auto-truncate prompt to nearest multiple of world_size so the
            # contiguous sequence shard math works regardless of caller's
            # exact tokenization. (Tokenizers can produce off-by-a-few token
            # counts that don't divide evenly — happens regularly with
            # synthetic prompts that target a specific token count.)
            if S % self._world_size != 0:
                S_aligned = (S // self._world_size) * self._world_size
                if self._rank == 0:
                    logger.warning(
                        "context_length=%d not divisible by world_size=%d; "
                        "truncating to %d for contiguous sequence sharding",
                        S, self._world_size, S_aligned,
                    )
                input_ids = input_ids[:, :S_aligned].contiguous()
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :S_aligned].contiguous()
                S = S_aligned
            S_local = S // self._world_size
            start = self._rank * S_local
            end   = start + S_local
            local_ids = input_ids[:, start:end].contiguous()
            local_pos = torch.arange(start, end, device=device).unsqueeze(0).expand(B, -1)
            # attention_mask under sharding is implicit (causal handled by ring kernel)
            local_attn_mask = None
        else:
            local_ids = input_ids
            local_pos = None
            local_attn_mask = attention_mask

        # M4 C11 (true NF4 storage): when cfg.kv_quant=True, supply
        # an empty NF4DynamicCache as the initial past_key_values so
        # HF's LlamaModel uses it instead of constructing its own
        # bf16 DynamicCache. Every patched LlamaAttention layer's
        # update() call then routes through NF4 quantization at
        # append time. The cache object is mutated in place across
        # the rest of the verify loop, so we keep a single instance
        # for the duration of generate().
        initial_cache = None
        if cfg.kv_quant:
            from src.models.nf4_dynamic_cache import NF4DynamicCache
            # Outlier-keep: only the rank holding global position 0
            # (rank 0 under sequence-parallel sharding) gets a bf16
            # prefix. Other ranks' caches are pure NF4. The prefix
            # protects the first ~128 tokens (attention sinks per
            # StreamingLLM) which are disproportionately attended to
            # and account for most of the NF4 acceptance loss.
            prefix_size = (
                cfg.kv_outlier_prefix_size if self._rank == 0 else 0
            )
            initial_cache = NF4DynamicCache(
                block_size=cfg.kv_block_size_nf4,
                dtype=cfg.torch_dtype,
                bf16_prefix_size=prefix_size,
                update_chunk_size=cfg.nf4_update_chunk_size,
                # Target-only decode appends ONE token per step; without the
                # fold the layer accumulates one chunk per step and every
                # forward concatenates all of them (R1). Exact, not
                # approximate -- see NF4DynamicCache._append_nf4_chunk.
                tail_merge_below=cfg.nf4_tail_merge_below,
            )

        # M4 Phase C 2026-05-10 lever #1: only the LAST position's
        # logits are used downstream (`local_last_logit = ...[:, -1, :]`).
        # Pass num_logits_to_keep=1 so HF only materializes the final
        # token's logits row instead of the full (B, S_local, vocab)
        # tensor. At 1M S_local=128k that's
        #   128k * 32000 * 2 = 8.2 GB saved per rank;
        # at 512k it's 4 GB. LlamaForCausalLM gained this kwarg in
        # transformers 4.45+; we bumped requirements-lock.txt from
        # 4.44.2 to 4.46.3 specifically to enable this.
        # Per-layer memory snapshots during prefill (paper Figure 3
        # + OOM attribution). Register forward hooks ONLY when
        # mem_tracer is active so M3 byte-identical replay isn't
        # affected. Hooks fire after each LlamaDecoderLayer's
        # forward and emit a labelled snapshot with layer_idx.
        # Removed before any decode/verify forward so verify-loop
        # forwards aren't spammed with hooks.
        prefill_hook_handles = []
        if mem_tracer is not None:
            try:
                layers = self.target_model.model.layers
                for layer_idx, layer in enumerate(layers):
                    def _make_hook(idx):
                        def _hook(module, inputs, output):
                            mem_tracer.snapshot(
                                f"prefill_after_layer_{idx:02d}",
                                layer_idx=idx,
                            )
                        return _hook
                    prefill_hook_handles.append(
                        layer.register_forward_hook(_make_hook(layer_idx))
                    )
            except Exception:
                # Best-effort. If the model doesn't expose .model.layers
                # in the expected shape, just skip per-layer snapshots —
                # the post_target_prefill snapshot still fires.
                prefill_hook_handles = []

        _nvtx_push("phase:prefill_target")
        with torch.cuda.stream(self.stream_compute):
            target_out = self.target_model(
                local_ids,
                attention_mask=local_attn_mask,
                position_ids=local_pos,
                use_cache=True,
                past_key_values=initial_cache,
                num_logits_to_keep=1,
            )
            past_kv          = target_out.past_key_values
            local_last_logit = target_out.logits[:, -1, :]
        self.stream_compute.synchronize()
        _nvtx_pop()

        # Always remove the per-layer hooks before decode. Otherwise
        # every verify round would re-fire 32 hook callbacks, each
        # writing JSON — measurable overhead and noise.
        for h in prefill_hook_handles:
            try:
                h.remove()
            except Exception:
                pass

        # Freeze the prefill boundary on every patched attention module so
        # subsequent decode forwards know where the sharded prefill ends and
        # the replicated tail begins.
        prefill_len = local_ids.shape[1]
        if self._world_size > 1:
            from src.models.ring_llama_attention import set_prefill_len
            set_prefill_len(self.target_model, prefill_len=prefill_len)

        # The "first generated token" is sampled from the LAST GLOBAL position's
        # logits, which only rank world_size-1 holds. Broadcast it so every rank
        # samples the same cur_token (deterministic given same seed + same RNG).
        if self._world_size > 1:
            dist.broadcast(local_last_logit, src=self._world_size - 1)
        next_token_logit = local_last_logit
        print(f"[TRACE rank={self._rank}] target prefill done, past_kv layers={len(past_kv)}", flush=True)
        if mem_tracer is not None:
            mem_tracer.snapshot("post_target_prefill")

        if cfg.debug:
            logger.debug("[RASD] prefill done, S=%d", S)

        # Draft prefill (skipped in target-only baseline mode).
        draft_past_kv = None
        if cfg.spec_steps > 0:
            # Prefill draft model — same tokenizer/vocab as target
            # (LLaMA-2 SentencePiece, vocab=32000). The draft sees the
            # leading BOS plus the most recent `draft_max_len - 1` tokens
            # (Sheared-LLaMA=4096, TinyLlama=2048, capped by
            # cfg.draft_window_cap): recency is the point, and a draft
            # conditioned on the document's opening proposes tokens the
            # target then rejects for context it never had.
            raw_draft_ids = draft_input_ids if draft_input_ids is not None else input_ids
            draft_ids = _draft_window(raw_draft_ids, self.draft_max_len)
            logger.info("[draft-window] draft prefill over %d of %d prompt tokens (window=%d)",
                        draft_ids.shape[1], raw_draft_ids.shape[1], self.draft_max_len)
            print(f"[TRACE rank={self._rank}] calling draft prefill, draft_S={draft_ids.shape[1]}", flush=True)
            _nvtx_push("phase:prefill_draft")
            with torch.cuda.stream(self.stream_draft):
                draft_out = self.draft_model(draft_ids, use_cache=True)
                draft_past_kv = draft_out.past_key_values
            self.stream_draft.synchronize()
            _nvtx_pop()
            print(f"[TRACE rank={self._rank}] draft prefill done", flush=True)
            if mem_tracer is not None:
                mem_tracer.snapshot("post_draft_prefill")
        else:
            print(f"[TRACE rank={self._rank}] target-only mode — skipping draft prefill", flush=True)

        if cfg.debug:
            pass

        # Seed the first generated token from target prefill (target-vocab safe)
        cur_token = _sample(next_token_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)  # (B,1)
        generated  = [cur_token]
        # The seed token is the FIRST emitted position, so its gap goes at the
        # front of the list: `token_gaps` is aligned with `generated`, and
        # both consumers (the sidecar length check and the tie rule's
        # position lookup) read it as a parallel array. Recorded HERE, above
        # the speculative/target-only split, so the two arms cannot drift.
        seed_gap: List[float] = (
            _step_gap(next_token_logit) if int(self._rank) == 0 else [])

        # TTFT (C12, mentor M4 metric): time from generate() entry to the
        # first output token being sampled. Captures prefill cost (target +
        # draft + first-token broadcast under multi-rank) but excludes the
        # speculative verify loop. All ranks lockstep on cur_token sample;
        # rank 0's reading is representative.
        torch.cuda.synchronize()
        t_first_token = time.perf_counter()

        # Global sequence length — needed under multi-rank to compute correct
        # position_ids for each verify forward. After prefill of S prompt
        # tokens, global_seqlen = S. The seed `cur_token` is at position S,
        # the first verified token will be at position S+1, etc.
        global_seqlen = S

        # Tracking
        total_accepted   = 0
        total_draft_toks = 0
        n_rounds         = 0
        # Acceptance is measured over NON-TRUNCATED rounds only. A round cut
        # short by the budget had its accepted prefix verified but only
        # partially emitted, so its accepted/gamma is not a draw from the
        # same distribution as the others: keeping it in the mean biases the
        # primary metric, and it biases it in the direction of the observed
        # result rather than randomly.
        acc_rounds       = 0     # rounds included in the acceptance mean
        acc_verified     = 0     # sum of n_acc over those rounds
        n_truncated      = 0
        # C13 sidecar (gated by cfg.log_per_token; cheap when disabled)
        per_token_trace: List[Dict] = []
        # Top-1 minus top-2 logit gap at every emitted position, in emission
        # order, so it lines up element-for-element with `generated` (the
        # losslessness tie rule needs the gap at the position where two runs
        # first disagree). Rank 0 only, like the token ids themselves.
        # Starts with the SEED token's gap: the seed is an emitted token and
        # the rounds below append theirs, so the list and the ids stay in
        # step from the first position.
        emitted_gaps: List[float] = list(seed_gap)
        # ---- Target-only autoregressive baseline (M4 Phase C 2026-05-10) ----
        # When cfg.spec_steps == 0, skip the whole speculative decoding
        # loop and run plain single-token autoregressive decode through
        # the target model. Same prefill, same NF4 cache, same ring
        # attention, same outlier-keep — only spec decoding is removed.
        # This is the apples-to-apples baseline that isolates the
        # contribution of the speculation loop while keeping every
        # other variable identical to RASD.
        if cfg.spec_steps == 0:
            while sum(t.shape[1] for t in generated) < cfg.max_new_tokens:
                _nvtx_push(f"phase:autoregressive_step_{n_rounds:03d}")
                # Single-token forward through target. Position is the
                # next global position past everything already generated.
                t_input = cur_token  # (B, 1)
                t_pos = torch.arange(
                    global_seqlen, global_seqlen + 1, device=device
                ).unsqueeze(0).expand(B, -1)
                with torch.cuda.stream(self.stream_compute):
                    t_out = self.target_model(
                        t_input,
                        past_key_values=past_kv,
                        position_ids=t_pos,
                        use_cache=True,
                    )
                    target_logit = t_out.logits[:, -1, :]
                    past_kv = t_out.past_key_values
                self.stream_compute.synchronize()

                # Broadcast logit so all ranks sample identically (same
                # invariant as the verify-loop's broadcast).
                if self._world_size > 1:
                    dist.broadcast(target_logit, src=0)

                cur_token = _sample(
                    target_logit, cfg.temperature, cfg.top_p,
                ).unsqueeze(-1)  # (B, 1)
                generated.append(cur_token)
                # The same measurement as the speculative arm's, through the same
                # helper: the target's own top-1 minus top-2 gap at the position
                # that produced this token. One entry per emitted token, so the
                # two arms' lists have the same length as their ids by
                # construction. NOTE this is the one place the target-only arm
                # can record it -- `target_logit` is the step's own distribution
                # and nothing else in this loop retains it.
                if int(self._rank) == 0:
                    emitted_gaps.extend(_step_gap(target_logit))
                global_seqlen += 1
                n_rounds += 1

                # Periodic checkpoint (same C6 contract as the spec path)
                self._maybe_save_checkpoint(
                    n_rounds=n_rounds, global_seqlen=global_seqlen,
                    total_accepted=0, total_draft_toks=0,
                    cur_token=cur_token, generated=generated,
                    past_kv=past_kv, draft_past_kv=None,
                    per_token_trace=per_token_trace, prefill_len=prefill_len,
                )

                _nvtx_pop()  # close autoregressive_step_NNN

                if (not cfg.ignore_eos) and (cur_token == self.tokenizer.eos_token_id).all():
                    break

            # Skip the spec-decoding loop entirely.
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            generated_ids = torch.cat([input_ids] + generated, dim=1)
            tokens_gen    = generated_ids.shape[1] - S
            elapsed       = t_end - t_start

            # Same measurement as the speculative path, from the same object and
            # for the same reason: the identity check compares it against the
            # prompt length and the sidecar's id count, so it must not be
            # computed FROM them.
            sequence_len = int(generated_ids.shape[1])

            metrics = {
                "tokens_generated":  tokens_gen,
                "sequence_tokens":   sequence_len,
                "time_sec":          elapsed,
                "throughput_tps":    tokens_gen / elapsed if elapsed > 0 else 0.0,
                "acceptance_rate":   0.0,  # no spec decoding
                "mean_latency_ms":   elapsed * 1000 / max(tokens_gen, 1),
                "ttft_ms":           (t_first_token - t_start) * 1000,
                "gpu_peak_mem_mb":   torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                "n_rounds":          n_rounds,
                "spec_steps":        0,
                "prefetch_depth":    cfg.prefetch_depth,
                "kv_block_size":     cfg.kv_block_size,
                "draft_model":       "",  # target-only — no draft loaded
                "target_model":      cfg.target_model_name,
                "seed":              cfg.seed,
            }
            if cfg.log_per_token:
                metrics["per_token_trace"] = (
                    per_token_trace if self._rank == 0 else None
                )
            # The precision this run ACTUALLY used, measured off the loaded
            # model and the live cache rather than read from the config. Outside
            # the `save_generated_tokens` guard because it is a property of the
            # RUN, not of the sidecar: a row whose config claimed 4-bit weights
            # while the loader silently skipped them (the CPU/MPS path warns and
            # does exactly that) would otherwise report the claim.
            metrics["numerics"] = numerics_report(self.target_model, past_kv)
            if cfg.save_generated_tokens and self._rank == 0:
                # Raw generated token IDs for the losslessness check. `input_ids`
                # is the FULL prompt on every rank (only `local_ids` is sharded),
                # so slicing past `input_ids.shape[1]` returns exactly the
                # replicated `generated` list and nothing of the prompt.
                metrics["generated_token_ids"] = generated_ids[
                    0, input_ids.shape[1]:].tolist()
                # ... and the ids the target was actually fed, BOS included. See
                # the speculative path.
                metrics["engine_input_ids"] = input_ids[0].tolist()
                # The target's top-1 minus top-2 logit gap at every emitted
                # position, aligned element-for-element with the ids above. The
                # losslessness tie rule reads the gap at the position where two
                # arms first disagree, so the two lists must stay in step.
                metrics["token_gaps"] = [float(g) for g in emitted_gaps]
            if mem_tracer is not None:
                mem_tracer.snapshot("end", n_rounds=n_rounds)
                sidecar_path = mem_tracer.write()
                if self._rank == 0 and sidecar_path is not None:
                    logger.info("[memory_trace] wrote %s", sidecar_path)
                    metrics["memory_attribution_mb"] = mem_tracer.attribution_summary()
            return generated_ids, metrics

        # ---- Main speculative decoding loop ----
        while sum(t.shape[1] for t in generated) < cfg.max_new_tokens:
            _nvtx_push(f"phase:verify_round_{n_rounds:03d}")

            # === ITERATION-BOUNDARY STREAM SYNC (R5) ===
            # End of previous round committed cur_token, past_kv, and (under
            # partial rejection) draft_past_kv on the default stream via
            # bonus_sample + _truncate_kv. stream_draft and stream_compute
            # are about to read those tensors, so they must explicitly wait
            # for default-stream commits. Cheap event-based waits; saves us
            # from the same async-race class that produced α=0.018 on bf16
            # in Check 4. No-op on the first iteration (default stream has
            # no pending work post-prefill-sync).
            self.stream_draft.wait_stream(torch.cuda.current_stream())
            self.stream_compute.wait_stream(torch.cuda.current_stream())

            # === R6 ASYNC-RING DRAIN (2026-05-06) ===
            # Multi-rank async ring (cfg.prefetch_depth >= 1) at W=8 was
            # observed to deadlock around 7-13 verify rounds — the failure
            # threshold varied with seed (max_new=32 PASSed at seed=42 but
            # FAILed at seed=123) suggesting a timing-sensitive race rather
            # than a count-based bug. The most likely culprit is cumulative
            # NCCL/CUDA-event state on stream_compute from many ring forwards;
            # an explicit synchronize between rounds drains pending P2P state
            # so each round starts clean. Cost: ~1ms per round for the host
            # block, negligible vs the per-round verify forward (~10ms-100ms).
            # Single-rank or sync-mode multi-rank: no-op (no in-flight work).
            if self._world_size > 1 and cfg.prefetch_depth >= 1:
                torch.cuda.synchronize()

            # === DRAFT PHASE (stream_draft) ===
            # Generate k tokens with the cheap draft model.
            draft_tokens  = []
            draft_logits_ = []
            draft_input   = cur_token

            with torch.cuda.stream(self.stream_draft):
                for _ in range(cfg.spec_steps):
                    d_out = self.draft_model(
                        draft_input,
                        past_key_values=draft_past_kv,
                        use_cache=True,
                    )
                    draft_past_kv  = d_out.past_key_values
                    d_logit        = d_out.logits[:, -1, :]
                    d_tok          = _sample(d_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)
                    draft_tokens.append(d_tok)
                    draft_logits_.append(d_logit)
                    draft_input    = d_tok
                draft_seq    = torch.cat(draft_tokens,  dim=1)   # (B, k)
                draft_logits = torch.stack(draft_logits_, dim=1) # (B, k, vocab)

            if cfg.debug:
                self.stream_draft.synchronize()
                logger.debug("[RASD] round=%d draft_tokens=%s", n_rounds, draft_seq[0].tolist())

            # === VERIFICATION PHASE (stream_compute) ===
            # Ring rotation now lives inside LlamaAttention.forward (R3 dual-cache);
            # there is no separate prefetcher block any more. The only required
            # cross-stream wait is between draft and compute streams.

            # Target verify reads draft_seq/draft_logits written on stream_draft.
            # Without this wait, fast kernels (bf16) launch the target forward
            # before the draft tokens are committed, causing async out-of-bounds
            # indexing inside the embedding layer. NF4's slower kernels masked
            # this by running draft to completion first.
            self.stream_compute.wait_stream(self.stream_draft)

            prior_target_len = past_kv[0][0].shape[2] if past_kv is not None else 0

            with torch.cuda.stream(self.stream_compute):
                # Packed verify: target sees [cur_token, draft_seq] in ONE forward
                # and returns k+1 logits. Spec-decoding requires conditioning on
                # the DRAFT's tokens, not the target's own argmax, so the
                # previous autoregressive loop (which fed t_logit.argmax back in)
                # was computing the wrong distribution at positions > 0. See
                # analysis/m3_post_analysis_plan.md Check 2 and
                # tests/test_verification_math.py for the spec.
                t_input = torch.cat([cur_token, draft_seq], dim=1)     # (B, k+1)
                # Under multi-rank, supply absolute position_ids so RoPE in
                # each ring-patched LlamaAttention layer uses correct positions.
                # cur_token sits at global_seqlen, draft_seq at global_seqlen+1..
                if self._world_size > 1:
                    t_pos = torch.arange(
                        global_seqlen, global_seqlen + t_input.shape[1],
                        device=device,
                    ).unsqueeze(0).expand(B, -1)
                else:
                    t_pos = None
                t_out   = self.target_model(
                    t_input,
                    past_key_values=past_kv,
                    position_ids=t_pos,
                    use_cache=True,
                )
                target_logits_v = t_out.logits                          # (B, k+1, vocab)
                post_verify_kv  = t_out.past_key_values

            # stream_draft must see updated past_kv before next round
            self.stream_draft.wait_stream(self.stream_compute)

            # Subsequent accept/reject, bonus sampling, and _truncate_kv run on
            # the default stream and read target_logits_v + post_verify_kv
            # (produced on stream_compute). Without this wait, default-stream
            # kernels would race the still-executing verify forward.
            torch.cuda.current_stream().wait_stream(self.stream_compute)

            if cfg.debug:
                self.stream_compute.synchronize()

            # === MULTI-RANK CONSENSUS (Fix2 2026-05-06) ===
            # Ring attention's online-softmax merges K/V slices in a
            # rank-DIFFERENT order (rank r processes K_r → K_{r-1} → ...).
            # Floating-point bf16 (and even fp32) merge is non-associative,
            # so target_logits_v has small numerical drift across ranks.
            # When accept_prob = p_target/p_draft is near a `r ~ U[0,1)`
            # threshold, ranks can flip independently → n_acc DIFFERS across
            # ranks → _truncate_kv produces different local cache sizes →
            # next round's ring P2P size mismatches → NCCL coalesced timeout
            # at SeqNum ~3500-3600 (root cause of R6 deadlock 2026-05-06).
            # Fix: broadcast target_logits_v and draft_logits from rank 0
            # so all ranks compute identical accept/reject, n_acc, cur_token.
            # Cost: ~1 MB broadcast per round. Negligible.
            _broadcast_round_state(draft_seq, target_logits_v, draft_logits,
                                   self._world_size)

            # === ACCEPT / REJECT ===
            accepted, n_acc = _acceptance_mask(
                draft_seq,
                target_logits_v,
                draft_logits,
                cfg.temperature,
            )

            # C13 per-position sidecar — cur_token sits at global_seqlen,
            # the k draft tokens span global_seqlen+1..global_seqlen+k.
            # n_rounds is still 0-indexed at this point (incremented next line).
            # `max_new_tokens` is a CAP, and a round yields up to n_acc + 1
            # tokens, so an untruncated final round would overshoot it. The
            # generated length would then be a function of how much the draft
            # happened to be accepted, not a fixed protocol: the
            # prompt + BOS + generated == context identity would be false, the
            # losslessness check (which compares a fixed-length generation
            # against a target-only run that stops exactly on the cap) would
            # report every speculative cell as incomplete, and throughput at
            # 1.000x vs 1.004x of the cap would not be the same measurement
            # across arms.
            budget = cfg.max_new_tokens - sum(t.shape[1] for t in generated)
            n_emit, committed, with_bonus, round_truncated = _round_commit_plan(
                budget, n_acc, cfg.spec_steps)

            # Target KV truncation: drop the verified-but-uncommitted tail and
            # the un-emitted positions of a truncated round. Done here, before
            # the trace record, so `kv_len_after` can be MEASURED from the cache
            # rather than recomputed from the arithmetic the truncation was asked
            # to perform (which would make the cap smoke's KV assertion a
            # restatement of this line).
            past_kv = _truncate_kv(post_verify_kv, prior_target_len + committed)
            kv_len_after = _kv_seq_len(past_kv)
            if kv_len_after != prior_target_len + committed:
                # Not fatal -- the loop continues from the cache, which is the
                # authority -- but the truncation did not do what the round
                # accounting believes, so it is reported rather than silently
                # averaged.
                logger.warning(
                    "[KV] round=%d truncate asked for %d, cache holds %d",
                    n_rounds, prior_target_len + committed, kv_len_after,
                )

            # The target's own indifference at every emitted position of THIS
            # round, on rank 0, on EVERY round -- not only the first, and not
            # only when the per-token trace is on. The list is aligned with
            # `generated`, so a round skipped here shifts every later gap by that
            # round's length and the tie rule then reads the wrong token's gap.
            # (This replaces an `n_rounds == 0` guard that was standing in for
            # "rank 0": it also silently dropped every round after the first.)
            round_gaps: List[float] = []
            if int(self._rank) == 0:
                round_gaps = _round_emitted_gaps(
                    target_logits_v, n_emit, n_acc, with_bonus)
                emitted_gaps.extend(round_gaps)

            if cfg.log_per_token:
                rec = _build_per_token_record(
                    round_idx=n_rounds,
                    global_pos_start=global_seqlen,
                    spec_steps=cfg.spec_steps,
                    n_acc=n_acc,
                    draft_seq=draft_seq,
                    accepted=accepted,
                )
                # What the TARGET verified (n_acc) and what was actually
                # committed (n_emit) differ only in a truncated final round.
                # Both are recorded so the acceptance mean can exclude that
                # round instead of silently averaging a partial one.
                rec["n_emitted"] = int(n_emit)
                rec["round_truncated"] = bool(round_truncated)
                # KV geometry. A truncated round commits ONLY the verified
                # prefix it actually emitted: no bonus, and nothing for the
                # verified-but-unemitted tail. Recorded so the cap smoke can
                # assert it rather than trust it.
                rec["n_committed"] = int(committed)
                rec["kv_len_before"] = int(prior_target_len)
                rec["kv_len_after"] = int(kv_len_after)
                rec["emitted_gaps"] = list(round_gaps)
                per_token_trace.append(rec)

            total_accepted   += n_emit
            total_draft_toks += cfg.spec_steps
            if round_truncated:
                n_truncated += 1
            else:
                acc_rounds   += 1
                acc_verified += n_acc
            n_rounds         += 1

            if mem_tracer is not None and n_rounds in (1, 2, 4, 8):
                # Snapshot at first few rounds — verify-loop steady-state
                # is established by round 4. Saves JSON size on long runs
                # (>100 rounds at 1M).
                mem_tracer.snapshot(f"post_verify_round_{n_rounds}")

            if cfg.debug:
                logger.debug("[RASD] round=%d accepted=%d/%d", n_rounds, n_acc, cfg.spec_steps)

            # --- Truncate target KV to committed length: prior + n_acc + 1 ---
            # post_verify_kv holds all k+1 verify positions. Only the first
            # n_acc draft tokens and the bonus are committed; the rest must
            # be dropped so the next round's cur_token arrives at the correct
            # positional offset.
            # (Target KV was truncated above, before the trace record was
            # built, so `kv_len_after` is the cache's own answer.)

            # Collect accepted tokens
            for i in range(n_emit):
                generated.append(draft_seq[:, i:i+1])

            # --- Bonus token ---
            # Full acceptance OR greedy: plain sample / argmax from target.
            # Partial rejection at temperature > 0: draw from the residual
            # distribution max(0, p_target - p_draft) normalized (Leviathan
            # et al. 2023). Sampling plain p_target here biases the next
            # draft context toward tokens the draft already preferred,
            # silently lowering acceptance on subsequent rounds.
            if n_acc == cfg.spec_steps or cfg.temperature == 0.0:
                bonus_logit = target_logits_v[:, n_acc, :]
                cur_token   = _sample(bonus_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)
            else:
                t_probs_row = F.softmax(target_logits_v[:, n_acc, :] / cfg.temperature, dim=-1)
                d_probs_row = F.softmax(draft_logits[:, n_acc, :]    / cfg.temperature, dim=-1)
                resid = torch.clamp(t_probs_row - d_probs_row, min=0.0)
                resid = resid / (resid.sum(-1, keepdim=True) + 1e-12)
                cur_token = torch.multinomial(resid, num_samples=1)
            # Skipped only when the budget ran out mid-round; the sampled token
            # is still produced so the truncation path leaves no half-updated
            # state, it simply is not part of the generation.
            if with_bonus:
                generated.append(cur_token)

            # --- Draft KV fix-up ---
            # Draft loop absorbed prior_d + k positions [cur_token_prev,
            # d_0..d_{k-2}]. Committed draft-visible tokens this round are
            # cur_token_prev + d_0..d_{n_acc-1} (the bonus becomes NEXT
            # round's cur_token, absorbed there).
            #   n_acc < k  : truncate off (k - 1 - n_acc) stale positions
            #   n_acc == k : we need d_{k-1} absorbed; run one extra forward
            if n_acc < cfg.spec_steps:
                draft_commit_len = draft_past_kv[0][0].shape[2] - (cfg.spec_steps - 1 - n_acc)
                draft_past_kv = _truncate_kv(draft_past_kv, draft_commit_len)
            else:
                with torch.cuda.stream(self.stream_draft):
                    catchup = self.draft_model(
                        draft_seq[:, -1:],
                        past_key_values=draft_past_kv,
                        use_cache=True,
                    )
                    draft_past_kv = catchup.past_key_values

            # Track global sequence length for next round's RoPE positions.
            # The verify committed `committed` new tokens to the global
            # context: the accepted prefix plus the bonus or resampled token.
            # In a truncated final round that is fewer than n_acc + 1, which is
            # exactly what keeps the total at the cap.
            global_seqlen += committed

            # ---- C6 SAVE: periodic checkpoint of verify-loop state ----
            # Gated on cfg.checkpoint_every > 0 (default 0 = disabled).
            # All ranks save their own KV slice (per-rank file path).
            self._maybe_save_checkpoint(
                n_rounds=n_rounds, global_seqlen=global_seqlen,
                total_accepted=total_accepted, total_draft_toks=total_draft_toks,
                cur_token=cur_token, generated=generated,
                past_kv=past_kv, draft_past_kv=draft_past_kv,
                per_token_trace=per_token_trace, prefill_len=prefill_len,
            )

            _nvtx_pop()  # close verify_round_NNN NVTX range

            # The cap is reached. `with_bonus` false means this round was cut
            # short by the budget, so there is nothing left to generate.
            if not with_bonus:
                break

            # Early stop on EOS (suppressed when B3 ignore_eos is set)
            if (not cfg.ignore_eos) and (cur_token == self.tokenizer.eos_token_id).all():
                # Record that THIS round is the one that ended generation, so
                # acceptance can be split at the EOS boundary instead of at an
                # arbitrary token count. Set here rather than at append time
                # because the bonus token is sampled after the record is
                # appended (see _build_per_token_record).
                if cfg.log_per_token and per_token_trace:
                    per_token_trace[-1]["ended_on_eos"] = True
                break

        # ---- Finalize ----
        # Ring rotation now lives inside the attention forward, so there is
        # no out-of-band P2P state to drain or "dummy P2P rounds" to run for
        # peers. All ranks executed identical generate() loops in lockstep.
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        generated_ids = torch.cat([input_ids] + generated, dim=1)
        tokens_gen    = generated_ids.shape[1] - S
        elapsed       = t_end - t_start

        # THE SEQUENCE THE ENGINE ACTUALLY HOLDS, measured off the tensor rather
        # than reconstructed from its parts. `generated_ids` is the final
        # sequence: the prompt it was given, the leading BOS it prepended, and
        # every token it emitted. Recording `prompt + 1 + generated` here would
        # make the row's sequence a RESTATEMENT of the numbers that are supposed
        # to check it, and the identity check on every stage would then be a
        # tautology: it could not fail, so it would verify nothing. Read from the
        # tensor, the identity becomes a real cross-check between three
        # independently obtained numbers -- the prompt length (from the
        # tokenizer), the emitted ids (from the sidecar) and this length (from
        # the engine). Measured on every rank`, identical on every rank.
        sequence_len = int(generated_ids.shape[1])

        metrics = {
            "tokens_generated":  tokens_gen,
            "sequence_tokens":   sequence_len,
            "time_sec":          elapsed,
            "throughput_tps":    tokens_gen / elapsed if elapsed > 0 else 0.0,
            # == mean over non-truncated rounds of (n_acc / gamma), which is
            # exactly `alpha_round` as the analysis modules compute it. The
            # previous numerator summed n_emit, which is n_acc only for
            # untruncated rounds, so the CSV and the cluster bootstrap
            # disagreed on the same run.
            "acceptance_rate":   (acc_verified / (acc_rounds * cfg.spec_steps)
                                  if acc_rounds > 0 else 0.0),
            "acceptance_rounds": acc_rounds,
            "rounds_excluded_truncated": n_truncated,
            "mean_latency_ms":   elapsed * 1000 / max(tokens_gen, 1),
            "ttft_ms":           (t_first_token - t_start) * 1000,
            "gpu_peak_mem_mb":   torch.cuda.max_memory_allocated(device) / 1024 ** 2,
            "n_rounds":          n_rounds,
            "spec_steps":        cfg.spec_steps,
            "prefetch_depth":    cfg.prefetch_depth,
            "kv_block_size":     cfg.kv_block_size,
            "draft_model":       cfg.draft_model_name,
            "target_model":      cfg.target_model_name,
            "seed":              cfg.seed,
        }
        if cfg.log_per_token:
            # Only rank 0 returns the trace to avoid duplicate sidecars;
            # all ranks lockstep so traces are identical anyway.
            metrics["per_token_trace"] = (
                per_token_trace if self._rank == 0 else None
            )
        # See the target-only path: measured, not read from the config, and
        # recorded outside the sidecar guard because it is a property of the run.
        metrics["numerics"] = numerics_report(self.target_model, past_kv)
        if cfg.save_generated_tokens and self._rank == 0:
            # Raw generated token IDs for the losslessness check. `input_ids` is
            # the FULL prompt on every rank (only `local_ids` is sharded), so
            # this slice returns exactly `generated` (see the target-only path).
            metrics["generated_token_ids"] = generated_ids[
                0, input_ids.shape[1]:].tolist()
            # The ids the TARGET was actually fed, INCLUDING the leading BOS.
            # Anything that claims to compare against "the same prompt" -- the
            # vLLM baseline in particular -- has to be given these, because the
            # BOS is part of what the target conditioned on: a comparison against
            # the prompt ids without it is a comparison against a different
            # sequence, and one token of difference at the front changes every
            # position's rotary phase.
            metrics["engine_input_ids"] = input_ids[0].tolist()
            # Same measurement as the target-only arm's, in the same order:
            # one gap per emitted token, aligned with the ids above.
            metrics["token_gaps"] = [float(g) for g in emitted_gaps]

        if mem_tracer is not None:
            mem_tracer.snapshot("end", n_rounds=n_rounds)
            sidecar_path = mem_tracer.write()
            if self._rank == 0 and sidecar_path is not None:
                logger.info("[memory_trace] wrote %s", sidecar_path)
                # Surface the per-stage attribution in the rank-0 metrics
                metrics["memory_attribution_mb"] = mem_tracer.attribution_summary()

        return generated_ids, metrics

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, **kwargs) -> Tuple[str, Dict]:
        """String-in, string-out wrapper around generate().

        Draft and target share the same LLaMA-2 tokenizer (vocab=32000).
        We still truncate the draft input to the draft model's max sequence
        length (TinyLlama=2048, Sheared-LLaMA=4096).
        """
        device = self._device
        target_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        # `draft_input_ids` stays None: the draft window is enforced in ONE
        # place, `generate`, as the leading BOS + the most recent
        # (`draft_max_len` - 1) tokens. Building a second, separately truncated
        # draft input here is how the two definitions of "the draft's context"
        # came apart -- this call used HF's default right-truncation, which keeps
        # the FIRST `draft_max_len` tokens, so at every context above the window
        # the draft was conditioned on the opening of the book while the target
        # was conditioned on its ending.
        out_ids, metrics = self.generate(
            target_inputs["input_ids"],
            attention_mask=target_inputs.get("attention_mask"),
            draft_input_ids=None,
            **kwargs,
        )
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return text, metrics

    def terminate(self):
        """Terminate the RunPod pod (calls RunPod API). No-op if not on RunPod."""
        pod_id = os.environ.get("RUNPOD_POD_ID")
        if pod_id:
            import urllib.request, json
            api_key = os.environ.get("RUNPOD_API_KEY", "")
            query   = f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}'
            req     = urllib.request.Request(
                f"https://api.runpod.io/graphql?api_key={api_key}",
                data=json.dumps({"query": query}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req)
            logger.info("Pod %s terminated.", pod_id)
