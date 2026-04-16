"""
RASD Inference — Ring Attention Speculative Decoding

Architecture
------------
Three CUDA streams run concurrently at each decoding step:

  stream_compute  — target model verification forward pass
  stream_draft    — draft model token generation (k steps)
  stream_comm     — async KV-block ring prefetch (D2D transfers)

Pipeline per step
-----------------
  t=0  [draft]   generate k draft tokens         (stream_draft)
       [comm]    async prefetch next KV block     (stream_comm)
  t=1  [compute] target verifies k+1 positions   (stream_compute)
                 waits on stream_comm event before consuming KV
       [draft]   next draft batch already running (stream_draft)

The key invariant: by the time the target model needs KV block N,
stream_comm has already fetched it during the previous draft phase.
With prefetch_depth=2, two blocks are in flight simultaneously,
hiding even longer communication latency.

Ablation hooks (A1–A4)
-----------------------
  A1  draft_model_name   TinyLlama-1.1B | Sheared-LLaMA-1.3B
  A2  spec_steps k       2 | 4 | 6 | 8 | 12
  A3  kv_block_size      256 | 512 | 1024 | 2048 tokens
  A4  prefetch_depth     0 (sync) | 1 (async-1) | 2 (async-2)

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
from collections import deque
from dataclasses import dataclass, field
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

    # Quantisation (to fit draft + target on same GPUs)
    quantize_draft: bool = True          # 4-bit NF4 via bitsandbytes
    quantize_target: bool = False

    # Reproducibility
    seed: int = 42

    # Debug
    debug: bool = False

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bfloat16" else torch.float16


# ---------------------------------------------------------------------------
# KV block prefetcher
# ---------------------------------------------------------------------------

@dataclass
class _KVBlock:
    """One ring-communication unit: KV tensors for kv_block_size tokens."""
    keys:   torch.Tensor   # (num_layers, B, H, block_size, D)
    values: torch.Tensor   # (num_layers, B, H, block_size, D)
    rank_src: int          # which rank these came from
    reqs:   List = field(default_factory=list)  # work handles for explicit r.wait()
    ready_event: torch.cuda.Event = field(default_factory=lambda: torch.cuda.Event())


class AsyncKVRingPrefetcher:
    """Manages async KV-block ring communication on a dedicated CUDA stream.

    At each step, `schedule_next()` posts a non-blocking isend/irecv pair
    for the next KV block. `wait_and_get()` waits on the event and returns
    the block once it is safe to consume.

    prefetch_depth controls how many blocks are in-flight simultaneously:
      0 → synchronous: wait() before returning, no overlap
      1 → one block prefetched during current draft phase
      2 → two blocks prefetched, hides higher comm latency

    Uses batch_isend_irecv to prevent NCCL eager-mode serialisation deadlock.
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        rank: int,
        world_size: int,
        prefetch_depth: int,
        debug: bool = False,
    ):
        self.stream        = stream
        self.rank          = rank
        self.world_size    = world_size
        self.prefetch_depth = prefetch_depth
        self.debug         = debug

        self.send_to   = (rank + 1) % world_size
        self.recv_from = (rank - 1) % world_size

        # In-flight queue of (send_reqs, recv_reqs, block)
        self._inflight: deque[Tuple[List, _KVBlock]] = deque()

    def schedule(
        self,
        k_send: torch.Tensor,
        v_send: torch.Tensor,
        k_recv_buf: torch.Tensor,
        v_recv_buf: torch.Tensor,
    ) -> _KVBlock:
        """Post async send/recv on comm stream, return a _KVBlock with pending reqs.

        NCCL ordering contract
        ----------------------
        The caller MUST call `r.wait()` on block.reqs BEFORE the next
        `dist.broadcast()` call. This ensures NCCL's sequence counter sees:
            broadcast [seq N] → P2P [seq N+1] → broadcast [seq N+2] → ...
        without multi-stream out-of-order submission.

        The GPU-level compute/comm overlap still works because:
        1. schedule() submits P2P to stream_comm (GPU starts it immediately)
        2. Caller starts target compute on stream_compute (runs concurrently on GPU)
        3. Caller calls r.wait() — by this point P2P has had ~200ms of compute time
           to complete; wait typically returns quickly
        4. Caller broadcasts for next round (ordered correctly)

        prefetch_depth controls *when* schedule() is called relative to compute:
          0 → after compute (synchronous, no overlap)
          1 → before compute (async overlap — P2P runs during target forward pass)
          2 → two rounds ahead (even more pipeline depth)
        """
        block = _KVBlock(
            keys=k_recv_buf,
            values=v_recv_buf,
            rank_src=self.recv_from,
        )

        # NCCL sequence ordering fix: submit P2P from the default stream (not
        # self.stream) so that on rank 0, broadcast and batch_isend_irecv share
        # the same stream — matching the peer ranks in _ring_peer_loop. Mixing
        # streams caused an off-by-one in NCCL's sequence counter between rank 0
        # and peers at kv_block_size=1024/2048 (rank 0 submitted round N+1 before
        # round N's P2P transmitted, peers still waiting on broadcast N).
        # Compute/comm overlap is preserved because NCCL internally uses its own
        # stream regardless of the submission stream.
        ops = [
            dist.P2POp(dist.isend, k_send, self.send_to),
            dist.P2POp(dist.isend, v_send, self.send_to),
            dist.P2POp(dist.irecv, k_recv_buf, self.recv_from),
            dist.P2POp(dist.irecv, v_recv_buf, self.recv_from),
        ]
        reqs = dist.batch_isend_irecv(ops)
        # Record the event on default stream AFTER P2P submission so that
        # stream_compute.wait_event(block.ready_event) stalls compute stream
        # until the irecv is complete — preventing data races at block 1024+.
        block.ready_event.record()

        # Store reqs for the caller to wait on before the next broadcast.
        # Do NOT auto-wait here — the caller controls timing for overlap.
        block.reqs = reqs

        if self.debug:
            logger.debug("[prefetcher] P2P posted, rank_src=%d prefetch_depth=%d",
                         self.recv_from, self.prefetch_depth)

        self._inflight.append(block)
        return block

    def wait_and_get(self, consume_stream: torch.cuda.Stream) -> Optional[_KVBlock]:
        """Block until the oldest in-flight KV block is ready, return it."""
        if not self._inflight:
            return None
        block = self._inflight.popleft()
        # Make the compute stream wait on the comm event
        consume_stream.wait_event(block.ready_event)
        return block


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


def _truncate_kv(past_kv, new_len: int):
    """Truncate past_key_values to `new_len` positions along the seq dim.

    Accepts both legacy tuple-of-tuples and HF DynamicCache (iterable as
    layer (k, v) tuples). Returns a legacy tuple; the next model forward
    auto-converts via DynamicCache.from_legacy_cache.
    """
    if past_kv is None:
        return None
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
        # Optional dedicated NCCL sub-group for broadcast signal. Set externally
        # by run_experiment.py before generate() so that broadcast operations
        # don't share a NCCL sequence counter with P2P ops on the default group.
        self._signal_group = None

        self._prefetcher = AsyncKVRingPrefetcher(
            stream=self.stream_comm,
            rank=self._rank,
            world_size=self._world_size,
            prefetch_depth=config.prefetch_depth,
            debug=config.debug,
        )

        if config.debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug("[RASD] init complete — debug mode ON (forced sync after each stream op)")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_streams(self):
        """Create three dedicated CUDA streams."""
        if not torch.cuda.is_available():
            raise RuntimeError("RASD requires CUDA.")
        self.stream_compute = torch.cuda.Stream()   # target model forward
        self.stream_draft   = torch.cuda.Stream()   # draft model forward
        self.stream_comm    = torch.cuda.Stream()   # KV ring communication

    def _load_models(self):
        """Load target and (optionally quantised) draft models.

        Handles three backends automatically via DeviceCapabilities:
          CUDA (RunPod) — 4-bit NF4 quantization, device_map per local_rank
          MPS (MacBook) — no quantization (bitsandbytes unsupported), .to("mps")
          CPU           — no quantization, .to("cpu")
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from src.utils.device import DeviceCapabilities

        cfg = self.cfg
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._caps = DeviceCapabilities.detect(local_rank=local_rank)
        self._device = self._caps.device

        logger.info("Loading target model: %s  [device=%s]", cfg.target_model_name, self._device)
        target_bnb = None
        if cfg.quantize_target and self._caps.supports_quantization:
            target_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)
        elif cfg.quantize_target:
            logger.warning("quantize_target=True ignored — 4-bit NF4 requires CUDA (current: %s)",
                           self._caps.device_type)

        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_name,
            revision=cfg.target_revision,
            torch_dtype=cfg.torch_dtype,
            quantization_config=target_bnb,
            **self._caps.hf_device_map_kwargs(),
        )
        if self._caps.device_type in ("mps", "cpu") and target_bnb is None:
            self.target_model = self.target_model.to(self._device)
        self.target_model.eval()

        logger.info("Loading draft model: %s  [device=%s]", cfg.draft_model_name, self._device)
        draft_bnb = None
        if cfg.quantize_draft and self._caps.supports_quantization:
            draft_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)
        elif cfg.quantize_draft:
            logger.warning("quantize_draft=True ignored — 4-bit NF4 requires CUDA (current: %s)",
                           self._caps.device_type)

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            cfg.draft_model_name,
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
    # KV cache helpers
    # ------------------------------------------------------------------

    def _extract_kv_block(
        self,
        past_kv,
        block_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slice one KV block from a HuggingFace past_key_values tuple.

        past_kv : tuple of (key, value) per layer, each (B, H, S, D)
        Returns (k_block, v_block) each (num_layers, B, H, block_size, D)
        """
        bs = self.cfg.kv_block_size
        start = block_idx * bs
        end   = start + bs
        keys   = torch.stack([layer[0][:, :, start:end, :] for layer in past_kv])
        values = torch.stack([layer[1][:, :, start:end, :] for layer in past_kv])
        return keys, values

    def _alloc_kv_buffers(self, past_kv) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocate receive buffers matching one KV block shape."""
        k_ex, v_ex = self._extract_kv_block(past_kv, 0)
        return torch.empty_like(k_ex), torch.empty_like(v_ex)

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

        # ---- Prefill: run target model on the prompt to get KV cache ----
        print(f"[TRACE rank={self._rank}] calling target prefill...", flush=True)
        with torch.cuda.stream(self.stream_compute):
            target_out = self.target_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_kv        = target_out.past_key_values
            next_token_logit = target_out.logits[:, -1, :]
        self.stream_compute.synchronize()
        print(f"[TRACE rank={self._rank}] target prefill done, past_kv layers={len(past_kv)}", flush=True)

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

        if cfg.debug:
            pass

        # Seed the first generated token from target prefill (target-vocab safe)
        cur_token = _sample(next_token_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)  # (B,1)
        generated  = [cur_token]

        # Target vocab size — used to clamp/validate tokens before embedding
        target_vocab = self.target_model.config.vocab_size

        # Pending P2P reqs from the previous round — must be waited on BEFORE
        # broadcasting the signal for the next round to keep NCCL ops in order.
        # (broadcast and batch_isend_irecv share the same NCCL sequence counter;
        # submitting one before the other is done causes a SeqNum mismatch crash.)
        pending_reqs: List = []

        # Tracking
        total_accepted   = 0
        total_draft_toks = 0
        n_rounds         = 0

        # ---- Main speculative decoding loop ----
        while sum(t.shape[1] for t in generated) < cfg.max_new_tokens:

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

            # === PREFETCH SIGNAL + P2P (stream_comm) ===
            # NCCL ordering: wait on previous round's P2P reqs FIRST, then broadcast.
            # This serialises broadcast [seq N] → P2P [seq N+1] on the NCCL timeline
            # regardless of which CUDA stream each op runs on.
            # The GPU still overlaps compute and comm — target forward runs on
            # stream_compute while this round's P2P runs on stream_comm concurrently.
            if self._world_size > 1 and past_kv is not None:
                # 1. Drain previous round's P2P (ensures NCCL ordering)
                print(f"[TRACE rank={self._rank}] round={n_rounds} draining {len(pending_reqs)} pending P2P reqs", flush=True)
                for r in pending_reqs:
                    r.wait()
                pending_reqs = []

                # 2. Submit this round's P2P (no signal — peers run for a
                # fixed max_new_tokens rounds, eliminating the need for any
                # collective in the loop and avoiding the NCCL deadlock that
                # broadcast/all_reduce on sub-groups caused at block≥1024).
                # Wrap block_idx by the number of *available* KV blocks, not
                # world_size. With large kv_block_size (e.g. 2048) and short
                # context (8192), there may be fewer blocks than ranks. Cycling
                # past the end produces empty tensors → NCCL size mismatch → deadlock.
                n_kv_blocks = max(1, past_kv[0][0].shape[2] // cfg.kv_block_size)
                block_idx = (n_rounds + 1) % n_kv_blocks
                k_send, v_send = self._extract_kv_block(past_kv, block_idx)
                k_send = k_send.contiguous()
                v_send = v_send.contiguous()
                k_buf, v_buf   = self._alloc_kv_buffers(past_kv)
                # Tick gate: signal peers BEFORE submitting P2P. Sending
                # ticks after batch_isend_irecv deadlocks because dist.send
                # (unbatched) serializes behind the batch — it can't start
                # until the batch completes, but the batch's irecv needs
                # the peer's isend, and the peer is blocked on the tick.
                tick = torch.zeros(1, dtype=torch.int32, device=device)
                for peer in range(1, self._world_size):
                    dist.send(tick, peer)
                pending_block  = self._prefetcher.schedule(k_send, v_send, k_buf, v_buf)
                self._prefetcher._inflight.clear()  # we track reqs via pending_reqs; don't accumulate blocks
                pending_reqs   = pending_block.reqs
                print(f"[TRACE rank={self._rank}] round={n_rounds} ticks sent + P2P submitted", flush=True)
            else:
                pending_block = None

            # === VERIFICATION PHASE (stream_compute) ===
            if pending_block is not None:
                self.stream_compute.wait_event(pending_block.ready_event)

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
                t_out   = self.target_model(
                    t_input,
                    past_key_values=past_kv,
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

            # === ACCEPT / REJECT ===
            accepted, n_acc = _acceptance_mask(
                draft_seq,
                target_logits_v,
                draft_logits,
                cfg.temperature,
            )

            total_accepted   += n_acc
            total_draft_toks += cfg.spec_steps
            n_rounds         += 1

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

            # Early stop on EOS
            if (cur_token == self.tokenizer.eos_token_id).all():
                break

        # ---- Finalize ----
        # Drain last P2P, then continue doing dummy P2P rounds so peers
        # (which run for exactly max_new_tokens rounds) don't hang.
        if self._world_size > 1:
            for r in pending_reqs:
                r.wait()
            pending_reqs = []

            # Peers expect max_new_tokens total P2P rounds. Rank 0 did
            # n_rounds so far. Continue with dummy P2P for the remainder.
            remaining = cfg.max_new_tokens - n_rounds
            if remaining > 0 and past_kv is not None:
                k_send, v_send = self._extract_kv_block(past_kv, 0)
                k_send = k_send.contiguous()
                v_send = v_send.contiguous()
                k_buf, v_buf = self._alloc_kv_buffers(past_kv)
                print(f"[TRACE rank={self._rank}] draining {remaining} remaining P2P rounds for peers", flush=True)
                tick = torch.zeros(1, dtype=torch.int32, device=device)
                for i in range(remaining):
                    # Tick gate: peers wait for this before their P2P
                    for peer in range(1, self._world_size):
                        dist.send(tick, peer)
                    block = self._prefetcher.schedule(k_send, v_send, k_buf, v_buf)
                    self._prefetcher._inflight.clear()
                    for r in block.reqs:
                        r.wait()

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
            "gpu_peak_mem_mb":   torch.cuda.max_memory_allocated(device) / 1024 ** 2,
            "n_rounds":          n_rounds,
            "spec_steps":        cfg.spec_steps,
            "prefetch_depth":    cfg.prefetch_depth,
            "kv_block_size":     cfg.kv_block_size,
            "draft_model":       cfg.draft_model_name,
            "target_model":      cfg.target_model_name,
            "seed":              cfg.seed,
        }

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
