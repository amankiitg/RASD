import math
import torch
import torch.nn as nn


class SlidingWindowAttention(nn.Module):
    """Sliding-window local attention baseline.

    Each token attends to a fixed-sized window of tokens around it.
    This naive implementation is simple and correct but not optimized for speed.
    """

    def __init__(self, dim, num_heads=8, window_size=2048, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S, head_dim)

        pads = self.window_size // 2
        k_padded = torch.nn.functional.pad(k, (0, 0, pads, pads), value=0.0)
        v_padded = torch.nn.functional.pad(v, (0, 0, pads, pads), value=0.0)

        outs = []
        for i in range(S):
            k_win = k_padded[:, :, i : i + self.window_size, :]
            v_win = v_padded[:, :, i : i + self.window_size, :]
            q_i = q[:, :, i : i + 1, :]
            scores = torch.einsum('b h i d, b h j d -> b h i j', q_i, k_win) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            out_i = torch.einsum('b h i j, b h j d -> b h i d', attn, v_win)
            outs.append(out_i)

        out = torch.cat(outs, dim=2)  # (B, H, S, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out(self.dropout(out))
        return out
