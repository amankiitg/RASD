import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


class RingAttention(nn.Module):
    """Ring Attention with distributed KV-cache exchange.

    When torch.distributed is initialized with world_size > 1, each process
    holds a local sequence shard (B, S_local, D). KV blocks rotate around the
    ring using point-to-point send/recv while Q stays fixed. An online softmax
    (log-sum-exp accumulator) ensures numerically stable combination of partial
    attention outputs across all ring steps.

    Falls back to blockwise local attention (still using online softmax for
    correctness) when running single-process or without a distributed backend.

    Args:
        dim: model hidden dimension
        num_heads: number of attention heads
        block_size: sequence block size used in the local (non-distributed) path
        dropout: attention dropout probability
    """

    def __init__(self, dim: int, num_heads: int = 8, block_size: int = 4096, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_distributed(self) -> bool:
        return (
            _DIST_AVAILABLE
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    @staticmethod
    def _online_softmax_update(out, lse, new_scores, v_block):
        """Numerically stable incremental softmax update.

        Given accumulated (out, lse) and a new block of scores/values,
        returns updated (out, lse).

        Args:
            out:       (B, H, S_q, head_dim) accumulated output so far
            lse:       (B, H, S_q) log-sum-exp so far
            new_scores:(B, H, S_q, S_kv) raw dot-product scores for this block
            v_block:   (B, H, S_kv, head_dim) values for this block

        Returns:
            updated out, updated lse
        """
        block_max = new_scores.amax(dim=-1)                         # (B,H,S_q)
        exp_s = torch.exp(new_scores - block_max.unsqueeze(-1))     # (B,H,S_q,S_kv)
        block_lse = block_max + torch.log(exp_s.sum(dim=-1))        # (B,H,S_q)

        new_lse = torch.logaddexp(lse, block_lse)                   # (B,H,S_q)
        scale_old = torch.exp(lse - new_lse).unsqueeze(-1)          # (B,H,S_q,1)
        scale_new = torch.exp(block_max - new_lse).unsqueeze(-1)    # (B,H,S_q,1)

        out = scale_old * out + scale_new * (exp_s @ v_block)       # (B,H,S_q,head_dim)
        return out, new_lse

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D) — full sequence (single-GPU) or local shard (multi-GPU)
        Returns:
            (B, S, D) attention output
        """
        if self._is_distributed():
            return self._ring_forward(x)
        return self._local_forward(x)

    # ------------------------------------------------------------------
    # Single-GPU path: blockwise attention with online softmax
    # ------------------------------------------------------------------

    def _local_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # Pad to multiple of block_size
        pad = (self.block_size - S % self.block_size) % self.block_size
        x_in = F.pad(x, (0, 0, 0, pad)) if pad else x
        S2 = x_in.shape[1]

        qkv = (
            self.qkv(x_in)
            .reshape(B, S2, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S2, head_dim)

        out = torch.zeros_like(q)
        lse = torch.full((B, self.num_heads, S2), float("-inf"), device=x.device, dtype=x.dtype)

        for j in range(0, S2, self.block_size):
            k_blk = k[:, :, j : j + self.block_size, :]
            v_blk = v[:, :, j : j + self.block_size, :]
            scores = torch.einsum("bhid,bhjd->bhij", q, k_blk) / math.sqrt(self.head_dim)
            out, lse = self._online_softmax_update(out, lse, scores, v_blk)

        # Remove padding, reshape, project
        out = out[:, :, :S, :].permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out(self.dropout(out))
        return out

    # ------------------------------------------------------------------
    # Multi-GPU path: true ring exchange
    # ------------------------------------------------------------------

    def _ring_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Distributed ring attention.

        Assumes the caller has already sharded the sequence across processes:
        each rank receives (B, S_local, D) where S_local = S_total / world_size.

        KV tensors rotate one step per ring iteration via async isend/irecv so
        that all ranks eventually attend over the full sequence.
        """
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size

        B, S_local, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, S_local, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S_local, head_dim)

        k_cur = k.contiguous()
        v_cur = v.contiguous()
        k_buf = torch.empty_like(k_cur)
        v_buf = torch.empty_like(v_cur)

        out = torch.zeros_like(q)
        lse = torch.full(
            (B, self.num_heads, S_local), float("-inf"), device=x.device, dtype=x.dtype
        )

        for step in range(world_size):
            scores = torch.einsum("bhid,bhjd->bhij", q, k_cur) / math.sqrt(self.head_dim)
            out, lse = self._online_softmax_update(out, lse, scores, v_cur)

            # Rotate KV ring: use batch_isend_irecv to post sends and recvs
            # atomically, avoiding the NCCL eager-mode serialization deadlock
            # that occurs when all ranks post isend before irecv.
            if step < world_size - 1:
                ops = [
                    dist.P2POp(dist.isend, k_cur, send_to),
                    dist.P2POp(dist.isend, v_cur, send_to),
                    dist.P2POp(dist.irecv, k_buf, recv_from),
                    dist.P2POp(dist.irecv, v_buf, recv_from),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
                # Swap buffers to avoid allocation
                k_cur, k_buf = k_buf, k_cur
                v_cur, v_buf = v_buf, v_cur

        out = out.permute(0, 2, 1, 3).reshape(B, S_local, D)
        out = self.out(self.dropout(out))
        return out
