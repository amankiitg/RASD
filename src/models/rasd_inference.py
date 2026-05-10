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
See M3_RING_INTEGRATION_PLAN.md open-question 1a for rationale.

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

    # Quantisation (to fit draft + target on same GPUs)
    quantize_draft: bool = True          # 4-bit NF4 via bitsandbytes
    quantize_target: bool = False

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
    """
    return {
        "round_idx":        int(round_idx),
        "global_pos_start": int(global_pos_start),
        "spec_steps":       int(spec_steps),
        "n_acc":            int(n_acc),
        "draft_tokens":     draft_seq[0].tolist(),
        "accepted":         [bool(x) for x in accepted[0].tolist()],
    }


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
                         rope_type: str = "linear"):
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

        `apply_rope_scaling` defaults to True (target). For the **draft**
        model we pass False: capping the draft at its native context cap
        (Sheared-LLaMA-1.3B = 4096) saves ~11 GB/rank of replicated KV at
        ctx=64k vs scaling the draft to match the target. Speculative
        decoding tolerates a smaller draft window — the
        `draft_ids = raw_draft_ids[:, -self.draft_max_len:]` truncation in
        generate() already gives the draft only the most recent native_max
        tokens of context. R6.4 OOM analysis (2026-05-06) showed draft KV
        was eating ~12 GB/rank at ctx=64k under the old behaviour.
        """
        from transformers import AutoConfig
        import math

        hf_cfg = AutoConfig.from_pretrained(model_name, revision=revision)
        if (apply_rope_scaling and context_length
                and context_length > hf_cfg.max_position_embeddings):
            native_max = hf_cfg.max_position_embeddings
            factor = float(math.ceil(context_length / native_max))
            hf_cfg.rope_scaling = _build_rope_scaling_dict(
                rope_type, factor, native_max,
            )
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
        )

        logger.info("Loading target model: %s  [device=%s]", cfg.target_model_name, self._device)
        target_bnb = None
        if cfg.quantize_target and self._caps.supports_quantization:
            target_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)
        elif cfg.quantize_target:
            logger.warning("quantize_target=True ignored — 4-bit NF4 requires CUDA (current: %s)",
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
        draft_bnb = None
        if cfg.quantize_draft and self._caps.supports_quantization:
            draft_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)
        elif cfg.quantize_draft:
            logger.warning("quantize_draft=True ignored — 4-bit NF4 requires CUDA (current: %s)",
                           self._caps.device_type)

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

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name, revision=cfg.target_revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_name, revision=cfg.draft_revision)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token

        # Max sequence length the draft model supports
        # TinyLlama=2048, Sheared-LLaMA=4096 — all share LLaMA-2 tokenizer
        self.draft_max_len = getattr(self.draft_model.config, "max_position_embeddings",
                             getattr(self.draft_model.config, "n_positions", 4096))

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
        if hasattr(draft_past_kv, "to_serializable"):
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

        # ---- C6 RESUME: try to load a checkpoint and skip prefill ----
        # Gated on cfg.checkpoint_every > 0 — when 0 (the default), this
        # whole branch is skipped and the M3 path is byte-identical.
        ckpt = self._try_load_checkpoint() if cfg.checkpoint_every > 0 else None

        # ---- Prefill: run target model on the prompt to get KV cache ----
        if ckpt is None:
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

            # Prefill draft model — same tokenizer/vocab as target (LLaMA-2 SentencePiece, vocab=32000).
            # Truncate to draft model's max sequence length if prompt is very long
            # (TinyLlama=2048, Sheared-LLaMA=4096). Keep last N tokens for recency.
            raw_draft_ids = draft_input_ids if draft_input_ids is not None else input_ids
            draft_ids = raw_draft_ids[:, -self.draft_max_len:]
            print(f"[TRACE rank={self._rank}] calling draft prefill, draft_S={draft_ids.shape[1]}", flush=True)
            with torch.cuda.stream(self.stream_draft):
                draft_out = self.draft_model(draft_ids, use_cache=True)
                draft_past_kv = draft_out.past_key_values
            self.stream_draft.synchronize()
            print(f"[TRACE rank={self._rank}] draft prefill done", flush=True)
            if mem_tracer is not None:
                mem_tracer.snapshot("post_draft_prefill")

            if cfg.debug:
                pass

            # Seed the first generated token from target prefill (target-vocab safe)
            cur_token = _sample(next_token_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)  # (B,1)
            generated  = [cur_token]

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
            # C13 sidecar (gated by cfg.log_per_token; cheap when disabled)
            per_token_trace: List[Dict] = []
        else:
            # ============================================================
            # C6 Resume — restore state, skip prefill, jump into the loop
            # ============================================================
            # If past_kv was serialized in NF4-native form (kv_quant=True
            # at save time), reconstruct an NF4DynamicCache; otherwise
            # the legacy bf16 tuple path is fine. (Fix for high-risk
            # finding #1 from 2026-05-10 third-pass review — without
            # NF4-native restore, NF4 storage would be permanently lost
            # after the first reload.)
            from src.models.nf4_dynamic_cache import (
                NF4DynamicCache as _NF4DC,
                is_nf4_serialized as _is_nf4,
            )
            if _is_nf4(ckpt.past_kv):
                past_kv = _NF4DC.from_serializable(ckpt.past_kv)
                if getattr(self, "_device", None) is not None:
                    past_kv.move_tensors_to(self._device)
            else:
                past_kv = ckpt.past_kv
            if _is_nf4(ckpt.draft_past_kv):
                draft_past_kv = _NF4DC.from_serializable(ckpt.draft_past_kv)
                if getattr(self, "_device", None) is not None:
                    draft_past_kv.move_tensors_to(self._device)
            else:
                draft_past_kv = ckpt.draft_past_kv
            cur_token        = ckpt.cur_token
            generated        = list(ckpt.generated)
            global_seqlen    = ckpt.global_seqlen
            n_rounds         = ckpt.n_rounds
            total_accepted   = ckpt.total_accepted
            total_draft_toks = ckpt.total_draft_toks
            per_token_trace  = list(ckpt.per_token_trace)
            prefill_len      = ckpt.prefill_len
            # Restore the patched ring attention's prefill boundary so the
            # next decode forward knows where sharded prefill ends.
            if self._world_size > 1:
                from src.models.ring_llama_attention import set_prefill_len
                set_prefill_len(self.target_model, prefill_len=prefill_len)
            # Recompute S from saved state — global_seqlen at checkpoint
            # time = S + (sum of per-round contributions). Total tokens
            # in `generated` = 1 (initial cur_token) + sum of contributions.
            generated_total = sum(t.shape[1] for t in generated)
            S = global_seqlen - (generated_total - 1)
            # Restore both CPU and CUDA RNG state. CPU alone is
            # insufficient when sampling runs on CUDA tensors (default
            # under multi-rank GPU): torch.multinomial / torch.rand_like
            # consume the device generator. (Fix for finding #3 from
            # 2026-05-10 review; original Fix #5 only addressed CPU.)
            if ckpt.rng_state is not None:
                torch.set_rng_state(ckpt.rng_state)
            if (ckpt.cuda_rng_state is not None
                    and torch.cuda.is_available()
                    and getattr(self, "_device", None) is not None
                    and self._device.type == "cuda"):
                torch.cuda.set_rng_state(ckpt.cuda_rng_state, self._device)
            # TTFT is meaningless on resume (we skipped prefill); record 0
            # so downstream metrics handling doesn't NaN.
            t_first_token = t_start
            logger.info(
                "[checkpoint] rank=%d resumed n_rounds=%d global_seqlen=%d "
                "(skipped prefill)", self._rank, n_rounds, global_seqlen,
            )

        # ---- Main speculative decoding loop ----
        while sum(t.shape[1] for t in generated) < cfg.max_new_tokens:

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
            if self._world_size > 1:
                dist.broadcast(target_logits_v, src=0)
                dist.broadcast(draft_logits, src=0)

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
            if cfg.log_per_token:
                per_token_trace.append(_build_per_token_record(
                    round_idx=n_rounds,
                    global_pos_start=global_seqlen,
                    spec_steps=cfg.spec_steps,
                    n_acc=n_acc,
                    draft_seq=draft_seq,
                    accepted=accepted,
                ))

            total_accepted   += n_acc
            total_draft_toks += cfg.spec_steps
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
            past_kv = _truncate_kv(post_verify_kv, prior_target_len + n_acc + 1)

            # Collect accepted tokens
            for i in range(n_acc):
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
            # The verify just committed (n_acc + 1) new tokens to the global
            # context: n_acc accepted draft tokens + 1 bonus or resampled token.
            global_seqlen += n_acc + 1

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

            # Early stop on EOS
            if (cur_token == self.tokenizer.eos_token_id).all():
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

        metrics = {
            "tokens_generated":  tokens_gen,
            "time_sec":          elapsed,
            "throughput_tps":    tokens_gen / elapsed if elapsed > 0 else 0.0,
            "acceptance_rate":   total_accepted / max(total_draft_toks, 1),
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
        # Truncate for draft model's positional embedding limit
        draft_inputs  = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.draft_max_len,
            truncation=True,
        ).to(device)
        out_ids, metrics = self.generate(
            target_inputs["input_ids"],
            attention_mask=target_inputs.get("attention_mask"),
            draft_input_ids=draft_inputs["input_ids"],
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
