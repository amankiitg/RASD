import math
import torch
import torch.nn as nn


class RingAttention(nn.Module):
    """Simplified ring/block attention.

    Splits sequence into equal-sized blocks and computes full attention inside each block.
    Additionally allows each block to attend to the next block (forming a ring).
    This is a straightforward, memory-friendly baseline for long-range sparsity.
    """

    def __init__(self, dim, num_heads=8, block_size=4096, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape
        device = x.device
        # pad to multiple of block_size
        pad = (self.block_size - (S % self.block_size)) % self.block_size
        if pad > 0:
            x = torch.cat([x, x.new_zeros((B, pad, D))], dim=1)
        S2 = x.shape[1]

        qkv = self.qkv(x).reshape(B, S2, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S2, head_dim)

        out_blocks = []
        for i in range(0, S2, self.block_size):
            q_block = q[:, :, i : i + self.block_size, :]
            k_block = k[:, :, i : i + self.block_size, :]
            v_block = v[:, :, i : i + self.block_size, :]

            # attend within block
            scores = torch.einsum('b h i d, b h j d -> b h i j', q_block, k_block)
            scores = scores / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)
            block_out = torch.einsum('b h i j, b h j d -> b h i d', attn, v_block)

            # also allow attending to next block (ring)
            j = (i + self.block_size) % S2
            k_next = k[:, :, j : j + self.block_size, :]
            v_next = v[:, :, j : j + self.block_size, :]
            scores2 = torch.einsum('b h i d, b h j d -> b h i j', q_block, k_next)
            scores2 = scores2 / math.sqrt(self.head_dim)
            attn2 = torch.softmax(scores2, dim=-1)
            block_out2 = torch.einsum('b h i j, b h j d -> b h i d', attn2, v_next)

            combined = block_out + block_out2
            out_blocks.append(combined)

        out = torch.cat(out_blocks, dim=2)[:, :, :S, :]
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out(self.dropout(out))
        return out
