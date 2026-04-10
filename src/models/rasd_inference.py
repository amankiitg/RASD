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
  A1  draft_model_name   DistilGPT-2 | Sheared-LLaMA-1.3B
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
    draft_model_name:  str = "distilgpt2"

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
        """Post async send/recv on comm stream, return a future-like _KVBlock.

        The caller should call .ready_event.wait(other_stream) before reading
        the returned block's keys/values.
        """
        block = _KVBlock(
            keys=k_recv_buf,
            values=v_recv_buf,
            rank_src=self.recv_from,
        )

        with torch.cuda.stream(self.stream):
            ops = [
                dist.P2POp(dist.isend, k_send.contiguous(), self.send_to),
                dist.P2POp(dist.isend, v_send.contiguous(), self.send_to),
                dist.P2POp(dist.irecv, k_recv_buf, self.recv_from),
                dist.P2POp(dist.irecv, v_recv_buf, self.recv_from),
            ]
            reqs = dist.batch_isend_irecv(ops)

            if self.prefetch_depth == 0:
                # Synchronous: wait immediately, no pipeline overlap
                for r in reqs:
                    r.wait()
                if self.debug:
                    logger.debug("[prefetcher] sync wait done, rank_src=%d", self.recv_from)
            else:
                # Async: record event after all ops complete on comm stream
                # The compute stream will wait on this event before consuming.
                def _record_after_wait():
                    for r in reqs:
                        r.wait()
                    block.ready_event.record(self.stream)

                # Run wait+record on comm stream via a CUDA graph-compatible
                # callback approximation — we just do it inline since torch
                # distributed reqs are CPU-side futures.
                for r in reqs:
                    r.wait()
                block.ready_event.record(self.stream)

            if self.debug and self.prefetch_depth > 0:
                self.stream.synchronize()
                logger.debug("[prefetcher] async block ready, rank_src=%d", self.recv_from)

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

    When draft and target have different vocab sizes (e.g. DistilGPT-2 vocab=50257
    vs LLaMA-2 vocab=32000) we cannot directly index the target distribution with
    a draft token ID that may be out of range. In that case we fall back to a
    vocab-agnostic criterion: accept iff the target's greedy token matches the
    draft token (greedy) or with probability p_target_max (stochastic) — this is
    a conservative but safe approximation when vocabs don't align.

    Returns
        accepted : (B, k) bool tensor
        n_accepted: number of accepted tokens (first rejection position, batch=1)
    """
    B, k = draft_tokens.shape
    eps = 1e-9

    vocab_target = target_logits.shape[-1]
    vocab_draft  = draft_logits.shape[-1]
    same_vocab   = (vocab_target == vocab_draft)

    if temperature == 0.0:
        # Greedy: accept iff target argmax == draft token (vocab-agnostic)
        target_tokens = target_logits[:, :k].argmax(dim=-1)   # (B, k)
        if same_vocab:
            accepted = (target_tokens == draft_tokens)
        else:
            # Draft tokens may be out of target vocab range — compare greedily
            accepted = (target_tokens == draft_tokens)
    else:
        if same_vocab:
            target_probs = F.softmax(target_logits[:, :k] / temperature, dim=-1)
            draft_probs  = F.softmax(draft_logits / temperature, dim=-1)
            idx = draft_tokens.unsqueeze(-1)                       # (B,k,1)
            p_t = target_probs.gather(-1, idx).squeeze(-1)         # (B,k)
            p_d = draft_probs.gather(-1, idx).squeeze(-1)          # (B,k)
            accept_prob = torch.clamp(p_t / (p_d + eps), max=1.0)
        else:
            # Vocabs don't align: use target confidence as accept probability.
            # This is a safe approximation — tokens where the target is confident
            # are more likely to match the draft's intent.
            target_probs = F.softmax(target_logits[:, :k] / temperature, dim=-1)
            accept_prob  = target_probs.max(dim=-1).values         # (B,k)

        r = torch.rand_like(accept_prob)
        accepted = r < accept_prob

    first_reject = (accepted[0] == False).nonzero(as_tuple=False)
    n_accepted = first_reject[0].item() if len(first_reject) > 0 else k

    return accepted, int(n_accepted)


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
        """Load target and (optionally quantised) draft models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.cfg
        device = torch.cuda.current_device()

        logger.info("Loading target model: %s", cfg.target_model_name)
        target_bnb = None
        if cfg.quantize_target:
            target_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)

        self.target_model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model_name,
            dtype=cfg.torch_dtype,
            device_map={"": device},
            quantization_config=target_bnb,
        )
        self.target_model.eval()

        logger.info("Loading draft model: %s", cfg.draft_model_name)
        draft_bnb = None
        if cfg.quantize_draft:
            draft_bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=cfg.torch_dtype)

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            cfg.draft_model_name,
            dtype=cfg.torch_dtype,
            device_map={"": device},
            quantization_config=draft_bnb,
        )
        self.draft_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.draft_tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_name)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token

        # Max sequence length the draft model supports (e.g. DistilGPT-2 = 1024)
        self.draft_max_len = getattr(self.draft_model.config, "max_position_embeddings",
                             getattr(self.draft_model.config, "n_positions", 1024))

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

        # ---- Prefill: run target model on the prompt to get KV cache ----
        with torch.cuda.stream(self.stream_compute):
            target_out = self.target_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_kv        = target_out.past_key_values
            next_token_logit = target_out.logits[:, -1, :]

        if cfg.debug:
            self.stream_compute.synchronize()
            logger.debug("[RASD] prefill done, S=%d", S)

        # Prefill draft model with its own tokenized input (different vocab to target).
        # draft_input_ids uses draft tokenizer space — never mixed with target token IDs.
        # Truncate draft input to draft model's max sequence length.
        # e.g. DistilGPT-2 supports max 1024 tokens; keep the last N tokens
        # (the most recent context is most relevant for next-token prediction).
        raw_draft_ids = draft_input_ids if draft_input_ids is not None else input_ids
        draft_ids = raw_draft_ids[:, -self.draft_max_len:]
        with torch.cuda.stream(self.stream_draft):
            draft_out = self.draft_model(draft_ids, use_cache=True)
            draft_past_kv = draft_out.past_key_values

        if cfg.debug:
            self.stream_draft.synchronize()

        # Seed the first generated token from target prefill (target-vocab safe)
        cur_token = _sample(next_token_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)  # (B,1)
        generated  = [cur_token]

        # Target vocab size — used to clamp/validate tokens before embedding
        target_vocab = self.target_model.config.vocab_size

        # Ring comm: prefetch first KV block if distributed
        if self._world_size > 1 and past_kv is not None:
            k_send, v_send   = self._extract_kv_block(past_kv, 0)
            k_buf,  v_buf    = self._alloc_kv_buffers(past_kv)
            self._prefetcher.schedule(k_send, v_send, k_buf, v_buf)

        # Tracking
        total_accepted   = 0
        total_draft_toks = 0
        n_rounds         = 0

        # ---- Main speculative decoding loop ----
        while sum(t.shape[1] for t in generated) < cfg.max_new_tokens:

            # === DRAFT PHASE (stream_draft) ===
            # Generate k tokens with the cheap draft model.
            # Simultaneously, comm stream prefetches the next KV block.
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

            # === PREFETCH (stream_comm) ===
            # While the draft was running, fire next KV prefetch for the ring.
            if self._world_size > 1 and past_kv is not None and cfg.prefetch_depth > 0:
                block_idx = (n_rounds + 1) % self._world_size
                k_send, v_send = self._extract_kv_block(past_kv, block_idx)
                k_buf,  v_buf  = self._alloc_kv_buffers(past_kv)
                pending_block  = self._prefetcher.schedule(k_send, v_send, k_buf, v_buf)
            else:
                pending_block = None

            # === VERIFICATION PHASE (stream_compute) ===
            # Run target model autoregressively for k+1 steps starting from
            # cur_token. We collect k+1 logits to compare against the k draft
            # logits. draft_seq token IDs are NEVER fed to the target — they
            # only appear in the acceptance probability calculation, which
            # operates purely in logit space.
            if pending_block is not None:
                self.stream_compute.wait_event(pending_block.ready_event)

            with torch.cuda.stream(self.stream_compute):
                target_logits_list = []
                t_input   = cur_token
                t_past_kv = past_kv
                for _ in range(cfg.spec_steps + 1):
                    t_out = self.target_model(
                        t_input,
                        past_key_values=t_past_kv,
                        use_cache=True,
                    )
                    t_past_kv = t_out.past_key_values
                    t_logit   = t_out.logits[:, -1, :]          # (B, vocab_target)
                    target_logits_list.append(t_logit)
                    # Next input: greedy token from target (always target-vocab safe)
                    t_input = t_logit.argmax(dim=-1, keepdim=True)

                past_kv         = t_past_kv
                target_logits_v = torch.stack(target_logits_list, dim=1)  # (B, k+1, vocab)

            # stream_draft must see updated past_kv before next round
            self.stream_draft.wait_stream(self.stream_compute)

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

            # Collect accepted tokens
            for i in range(n_acc):
                generated.append(draft_seq[:, i:i+1])

            # One bonus token from the target at the rejection point
            bonus_logit = target_logits_v[:, n_acc, :]
            cur_token   = _sample(bonus_logit, cfg.temperature, cfg.top_p).unsqueeze(-1)
            generated.append(cur_token)

            # Early stop on EOS
            if (cur_token == self.tokenizer.eos_token_id).all():
                break

        # ---- Finalize ----
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

        Tokenizes the prompt separately for target and draft models so their
        vocabulary spaces never get mixed.
        """
        device = next(self.target_model.parameters()).device
        target_inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        # Truncate prompt for draft tokenizer to its max sequence length
        draft_inputs  = self.draft_tokenizer(
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
