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
  [draft]   generate k draft tokens         (stream_draft)
  [verify]  target packed forward over [cur_token, draft_seq]
            with ring attention rotating sharded prefill K/V across ranks
            internally per layer; replicated tail handled locally.
                                            (stream_compute)
  [accept]  accept/reject + bonus sample    (default stream)

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
    # rope_scaling. We use linear interpolation (factor = ceil(ctx/native)).
    # 0 / None = no override (use the model's native max).
    context_length: int = 0

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
                         context_length: int, label: str):
        """Load the model's HF config and apply linear RoPE scaling if needed.

        Llama-2 ships with max_position_embeddings=4096. To run at longer
        contexts (e.g. 64k for the M3 ablation), we set rope_scaling with
        factor = ceil(ctx / native_max) so token positions get linearly
        interpolated into the trained RoPE range.
        """
        from transformers import AutoConfig
        import math

        hf_cfg = AutoConfig.from_pretrained(model_name, revision=revision)
        if context_length and context_length > hf_cfg.max_position_embeddings:
            native_max = hf_cfg.max_position_embeddings
            factor = float(math.ceil(context_length / native_max))
            hf_cfg.rope_scaling = {"type": "linear", "factor": factor}
            hf_cfg.max_position_embeddings = context_length
            logger.info(
                "[RoPE] %s: ctx=%d > native=%d → linear scaling factor=%.1f",
                label, context_length, native_max, factor,
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
            cfg.target_model_name, cfg.target_revision, cfg.context_length, label="target",
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
        )

        draft_hf_config = self._build_hf_config(
            cfg.draft_model_name, cfg.draft_revision, cfg.context_length, label="draft",
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
        # Under multi-rank (R3 dual-cache layout), each rank owns the contiguous
        # slice [rank*S/W, (rank+1)*S/W) of the prompt. Each rank embeds and
        # forwards its own slice; the patched LlamaAttention performs ring
        # attention across ranks for cross-slice attention. Position IDs are
        # the absolute global positions so RoPE produces correct embeddings.
        if self._world_size > 1:
            assert S % self._world_size == 0, (
                f"context_length={S} must be divisible by world_size={self._world_size} "
                f"for contiguous sequence sharding"
            )
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

        with torch.cuda.stream(self.stream_compute):
            target_out = self.target_model(
                local_ids,
                attention_mask=local_attn_mask,
                position_ids=local_pos,
                use_cache=True,
            )
            past_kv          = target_out.past_key_values
            local_last_logit = target_out.logits[:, -1, :]
        self.stream_compute.synchronize()

        # Freeze the prefill boundary on every patched attention module so
        # subsequent decode forwards know where the sharded prefill ends and
        # the replicated tail begins.
        if self._world_size > 1:
            from src.models.ring_llama_attention import set_prefill_len
            set_prefill_len(self.target_model, prefill_len=local_ids.shape[1])

        # The "first generated token" is sampled from the LAST GLOBAL position's
        # logits, which only rank world_size-1 holds. Broadcast it so every rank
        # samples the same cur_token (deterministic given same seed + same RNG).
        if self._world_size > 1:
            dist.broadcast(local_last_logit, src=self._world_size - 1)
        next_token_logit = local_last_logit
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

        # Global sequence length — needed under multi-rank to compute correct
        # position_ids for each verify forward. After prefill of S prompt
        # tokens, global_seqlen = S. The seed `cur_token` is at position S,
        # the first verified token will be at position S+1, etc.
        global_seqlen = S

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

            # Track global sequence length for next round's RoPE positions.
            # The verify just committed (n_acc + 1) new tokens to the global
            # context: n_acc accepted draft tokens + 1 bonus or resampled token.
            global_seqlen += n_acc + 1

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
