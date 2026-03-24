import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """Sliding-window local attention baseline.

    Each token attends to a fixed-size window of tokens centred on it.
    Uses tensor unfolding instead of a Python token-by-token loop, so it
    scales to 128 k+ sequence lengths without prohibitive Python overhead.

    Memory footprint is O(B * H * S * window_size) — keep window_size
    proportionally smaller than sequence length to stay within GPU memory.

    Args:
        dim: model hidden dimension
        num_heads: number of attention heads
        window_size: number of tokens each position attends to (must be even)
        dropout: attention dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert window_size % 2 == 0, "window_size must be even"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, D)
        Returns:
            (B, S, D)
        """
        B, S, D = x.shape
        w = self.window_size
        half_w = w // 2

        qkv = (
            self.qkv(x)
            .reshape(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S, head_dim)

        # Pad K and V so that token i has a full window [i-half_w, i+half_w).
        # left_pad = half_w, right_pad = half_w - 1  → padded length = S + w - 1
        # After unfold(size=w, step=1) we get exactly S windows.
        k_pad = F.pad(k, (0, 0, half_w, half_w - 1))   # (B, H, S+w-1, head_dim)
        v_pad = F.pad(v, (0, 0, half_w, half_w - 1))

        # unfold along seq dim: (B, H, S, head_dim, w) → permute → (B, H, S, w, head_dim)
        k_win = k_pad.unfold(2, w, 1).permute(0, 1, 2, 4, 3)  # (B, H, S, w, head_dim)
        v_win = v_pad.unfold(2, w, 1).permute(0, 1, 2, 4, 3)  # (B, H, S, w, head_dim)

        # Attention scores: q (B, H, S, head_dim) × k_win (B, H, S, w, head_dim)
        # → scores (B, H, S, w)
        scores = torch.einsum("bhid,bhiwd->bhiw", q, k_win) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum: attn (B, H, S, w) × v_win (B, H, S, w, head_dim) → (B, H, S, head_dim)
        out = torch.einsum("bhiw,bhiwd->bhid", attn, v_win)

        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out(out)
        return out
