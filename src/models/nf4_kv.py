"""NF4 quantization for K/V cache (M4 C11 — primary 1M memory lever).

NF4 (Normal Float 4-bit; Dettmers et al. 2023 "QLoRA") is a 4-bit
non-uniform quantization where the 16 codepoints are placed so that
quantiles of a standard normal distribution are evenly spread among
them. For zero-mean normally-distributed tensors (which K/V cache
activations approximately are after layer-norm + RoPE), NF4 minimizes
expected reconstruction error.

This module implements a **pure PyTorch** NF4 codec — CPU-compatible
for local testing, GPU-compatible for pod deployment. The pod's
production path can swap in bitsandbytes' CUDA kernels for ~10x
throughput, but the wire format and per-block scale convention here
are independent of bnb.

Storage layout:
- For a (..., N) input where N % block_size == 0, the codec produces:
    codes:  (..., N // 2)   uint8   — two 4-bit codes packed per byte
    scales: (..., N // block_size) float32 — per-block absmax scale

Memory savings vs bf16 storage:
  bf16:   2 bytes per element
  NF4:    0.5 bytes per element + (4 / block_size) bytes for scale
          = 0.5 + 0.0625 ≈ 0.56 bytes per element at block_size=64
  ratio:  ~3.6× compression

For the M4 1M ring KV budget (per-rank K/V at ctx=1M, W=8 = ~68 GB
in bf16), NF4 brings this to ~17 GB — fits comfortably under 40 GB
SXM2 with the rest of the per-rank memory budget.

Round-trip error:
  Per-block: expected L2 relative error < 5% on normal-distributed
  inputs. Empirical error on bf16 attention K/V activations is
  typically 1-2% (see KIVI/KVQuant prior art for comparable results).
"""
from __future__ import annotations

from typing import Tuple

import torch

# ---------------------------------------------------------------------------
# NF4 codepoints — Dettmers et al. 2023, exact values from QLoRA
# ---------------------------------------------------------------------------

NF4_CODES_FLOAT: Tuple[float, ...] = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)


def _codes_tensor(device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Codepoints as a tensor on the requested device — cached implicitly
    by the caller (we don't memoize because device might change)."""
    return torch.tensor(NF4_CODES_FLOAT, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_nf4(
    x: torch.Tensor, block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to NF4 codes + per-block scales.

    Args:
        x:         input tensor with last-dim divisible by block_size.
                   Any leading shape; quantization happens along the
                   last dim. dtype must be a floating type (bf16/fp16/fp32).
        block_size: number of elements per scale block. Standard is 64.

    Returns:
        codes:     uint8 tensor of shape (..., N // 2). Two 4-bit codes
                   packed per byte; the **lower** nibble is the code at
                   even index, the **upper** nibble is the code at odd
                   index. (Convention is consistent with bitsandbytes.)
        scales:    float32 tensor of shape (..., N // block_size). Per-
                   block absmax scale. Stored as fp32 so dequantization
                   has full precision; the bf16 multiply happens after
                   dequant when needed.

    The math: for each block, scale = max(abs(block)) (or 1.0 if zero).
    Each value is normalized: v_norm = v / scale (in [-1, 1]). The
    nearest of the 16 NF4 codepoints is chosen; its index is the 4-bit
    code. Reconstruction: code -> codepoint -> code * scale.
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"x.shape[-1]={x.shape[-1]} not divisible by block_size={block_size}"
        )
    if not x.is_floating_point():
        raise ValueError(f"x must be floating-point; got {x.dtype}")

    orig_shape = x.shape
    N = orig_shape[-1]
    num_blocks = N // block_size

    # Reshape so each block is on its own axis
    blocks = x.reshape(*orig_shape[:-1], num_blocks, block_size)

    # Per-block absmax. eps prevents divide-by-zero on all-zero blocks.
    scale = blocks.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(
        scale > 0, scale, torch.ones_like(scale)
    )

    # Normalize into [-1, 1]
    normed = blocks / scale  # same shape as blocks

    # Find nearest codepoint by squared distance.
    # codes_t shape (16,); broadcasting gives (..., num_blocks, block_size, 16)
    codes_t = _codes_tensor(x.device, dtype=normed.dtype)
    diffs = normed.unsqueeze(-1) - codes_t  # (..., num_blocks, block_size, 16)
    indices = diffs.abs().argmin(dim=-1).to(torch.uint8)  # (..., num_blocks, block_size)

    # Flatten the block dim back into the last axis: (..., N)
    indices = indices.reshape(*orig_shape)

    # Pack two 4-bit codes per byte (lower nibble = even idx, upper = odd)
    even = indices[..., 0::2]  # shape (..., N//2)
    odd  = indices[..., 1::2]
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)  # uint8 (..., N//2)

    # Squeeze the trailing keepdim from scale: (..., num_blocks, 1) -> (..., num_blocks)
    scales = scale.squeeze(-1).to(torch.float32)

    return packed, scales


def dequantize_nf4(
    codes: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 64,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of quantize_nf4. Returns a tensor with shape (..., N)
    where N = 2 * codes.shape[-1] and dtype = `dtype`.

    Args:
        codes:      uint8 packed codes from quantize_nf4
        scales:     float32 per-block scales from quantize_nf4
        block_size: must match the block_size used at quantize time
        dtype:      floating dtype for the output (bf16 / fp16 / fp32)
    """
    if codes.dtype != torch.uint8:
        raise ValueError(f"codes must be uint8; got {codes.dtype}")

    # Unpack lower + upper nibbles into separate tensors
    lower = (codes & 0x0F).to(torch.long)
    upper = ((codes >> 4) & 0x0F).to(torch.long)

    # Interleave back to the original index sequence
    interleaved = torch.stack([lower, upper], dim=-1).reshape(
        *codes.shape[:-1], codes.shape[-1] * 2
    )  # (..., N)

    N = interleaved.shape[-1]
    if N % block_size != 0:
        raise ValueError(
            f"unpacked length {N} not divisible by block_size={block_size}"
        )
    num_blocks = N // block_size

    codes_t = _codes_tensor(codes.device, dtype=dtype)
    values = codes_t[interleaved]  # (..., N) — gather codepoints

    # Apply per-block scale
    blocks = values.reshape(*values.shape[:-1], num_blocks, block_size)
    blocks = blocks * scales.unsqueeze(-1).to(dtype)
    return blocks.reshape(*values.shape).to(dtype)


# ---------------------------------------------------------------------------
# Memory accounting helper (informational — used by tests + paper)
# ---------------------------------------------------------------------------

def packed_size_bytes(num_elements: int, block_size: int = 64) -> int:
    """Bytes used by NF4 storage for `num_elements` floats.

    codes:  num_elements / 2 bytes (two 4-bit per byte)
    scales: (num_elements / block_size) * 4 bytes (fp32 scale)
    """
    if num_elements % block_size != 0:
        raise ValueError(
            f"num_elements={num_elements} not divisible by block_size={block_size}"
        )
    codes_bytes  = num_elements // 2
    scales_bytes = (num_elements // block_size) * 4
    return codes_bytes + scales_bytes


def compression_ratio(block_size: int = 64) -> float:
    """Bytes-per-element ratio bf16 -> NF4. ~3.6x at block_size=64."""
    bf16_bytes_per_elem = 2.0
    nf4_bytes_per_elem  = 0.5 + 4.0 / block_size  # 4 fp32 bytes amortized over block
    return bf16_bytes_per_elem / nf4_bytes_per_elem


# ---------------------------------------------------------------------------
# NF4 KV cache wrapper
# ---------------------------------------------------------------------------

class NF4KVCache:
    """Per-layer NF4-stored KV cache with append + read API.

    Wraps the dual-cache layout the M4 C11 plan calls for:
      [sharded_prefill_local | replicated_tail]

    Internally stores K and V as concatenated NF4 codes + per-block
    scales. Quantizes on append (after each verify round); dequantizes
    on read into bf16 SRAM tiles for the FA-2 kernel.

    Shape contract (matches HF legacy past_kv format):
      add_kv(k, v): k, v are (B, H, S_new, D) — append to position dim
      get_kv():     returns (k_dq, v_dq) of shape (B, H, S_total, D)
                    in `dtype` (default bf16)

    For the head_dim axis to quantize cleanly, head_dim must be
    divisible by block_size. Llama-2-7B has head_dim=128, so
    block_size=64 (default) works (2 blocks per head). Llama-2-13B
    has head_dim=128 too, same story.

    Single-rank scope: this wrapper handles a single rank's KV slice.
    Cross-rank rotation in the ring kernel uses pack_for_wire/
    unpack_from_wire (see below) to send NF4 packets — 4× less wire
    traffic. Rank-coordination happens in ring_attention_kernel.py.
    """

    def __init__(self, block_size: int = 64,
                 dtype: torch.dtype = torch.bfloat16):
        self.block_size = block_size
        self.dtype = dtype
        # Per-position lists of (codes, scales) tuples for K and V.
        # Each entry covers `block_size`-divisible chunks along head_dim.
        self._k_codes:  list[torch.Tensor] = []   # uint8
        self._k_scales: list[torch.Tensor] = []   # fp32
        self._v_codes:  list[torch.Tensor] = []
        self._v_scales: list[torch.Tensor] = []
        # Cached metadata for sanity checks
        self._B: int | None = None
        self._H: int | None = None
        self._D: int | None = None

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def add_kv(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Quantize and append a (B, H, S_new, D) k + v block."""
        if k.shape != v.shape:
            raise ValueError(f"k.shape={k.shape} != v.shape={v.shape}")
        if k.dim() != 4:
            raise ValueError(f"expect (B,H,S,D); got {tuple(k.shape)}")
        B, H, S_new, D = k.shape
        if D % self.block_size != 0:
            raise ValueError(
                f"head_dim={D} not divisible by block_size={self.block_size}"
            )
        # Lock dimensions on first append
        if self._B is None:
            self._B, self._H, self._D = B, H, D
        else:
            if (B, H, D) != (self._B, self._H, self._D):
                raise ValueError(
                    f"shape mismatch with previous appends: "
                    f"got ({B},{H},?,{D}); expected ({self._B},{self._H},?,{self._D})"
                )

        # Quantize along the last dim (head_dim) — natural for NF4 since
        # the head_dim is the dimension where activations are normally-
        # distributed after layer-norm + RoPE.
        k_codes, k_scales = quantize_nf4(k, block_size=self.block_size)
        v_codes, v_scales = quantize_nf4(v, block_size=self.block_size)

        self._k_codes.append(k_codes)
        self._k_scales.append(k_scales)
        self._v_codes.append(v_codes)
        self._v_scales.append(v_scales)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize all stored entries and concatenate along the position dim.

        Returns (k, v) of shape (B, H, S_total, D) in self.dtype.
        """
        if not self._k_codes:
            # Empty cache — return zero-length tensors with the right shape
            # if dimensions are known, else raise
            if self._B is None:
                raise RuntimeError(
                    "Cache is empty and dimensions not yet known — "
                    "call add_kv at least once before get_kv()"
                )
            empty = torch.empty(
                self._B, self._H, 0, self._D, dtype=self.dtype,
                device=self._k_codes[0].device if self._k_codes else "cpu",
            )
            return empty, empty.clone()

        ks = [
            dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
            for c, s in zip(self._k_codes, self._k_scales)
        ]
        vs = [
            dequantize_nf4(c, s, block_size=self.block_size, dtype=self.dtype)
            for c, s in zip(self._v_codes, self._v_scales)
        ]
        # Concatenate along position dim (axis 2 in (B, H, S, D))
        k_full = torch.cat(ks, dim=2)
        v_full = torch.cat(vs, dim=2)
        return k_full, v_full

    # ------------------------------------------------------------------
    # Truncation (for spec-decoding partial-rejection KV rollback)
    # ------------------------------------------------------------------

    def truncate(self, new_seqlen: int) -> None:
        """Keep only the first `new_seqlen` positions (drop the tail).

        Mirrors the _truncate_kv operation on bf16 past_kv. Block-by-
        block: drop entries whose cumulative length exceeds new_seqlen,
        then partially trim the boundary entry.
        """
        if new_seqlen < 0:
            raise ValueError(f"new_seqlen={new_seqlen} must be >= 0")
        cumulative = 0
        kept_k_codes:  list[torch.Tensor] = []
        kept_k_scales: list[torch.Tensor] = []
        kept_v_codes:  list[torch.Tensor] = []
        kept_v_scales: list[torch.Tensor] = []
        for kc, ks_, vc, vs_ in zip(
            self._k_codes, self._k_scales, self._v_codes, self._v_scales
        ):
            # Each entry stores S_chunk positions. Recover S_chunk from
            # the codes shape: codes is (B, H, S_chunk, D//2).
            chunk_seqlen = kc.shape[2]
            if cumulative + chunk_seqlen <= new_seqlen:
                kept_k_codes.append(kc)
                kept_k_scales.append(ks_)
                kept_v_codes.append(vc)
                kept_v_scales.append(vs_)
                cumulative += chunk_seqlen
            else:
                keep = new_seqlen - cumulative
                if keep > 0:
                    # Slice the chunk down to `keep` positions
                    kept_k_codes.append(kc[:, :, :keep, :].contiguous())
                    kept_k_scales.append(ks_[:, :, :keep, :].contiguous())
                    kept_v_codes.append(vc[:, :, :keep, :].contiguous())
                    kept_v_scales.append(vs_[:, :, :keep, :].contiguous())
                break
        self._k_codes  = kept_k_codes
        self._k_scales = kept_k_scales
        self._v_codes  = kept_v_codes
        self._v_scales = kept_v_scales

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total number of stored positions (sum of chunk lengths)."""
        return sum(c.shape[2] for c in self._k_codes)

    @property
    def num_chunks(self) -> int:
        return len(self._k_codes)

    def memory_bytes(self) -> int:
        """Total bytes stored on the device for K + V codes + scales."""
        total = 0
        for c, s in zip(self._k_codes, self._k_scales):
            total += c.numel() + s.numel() * 4
        for c, s in zip(self._v_codes, self._v_scales):
            total += c.numel() + s.numel() * 4
        return total
