"""Memory-mapped dataset loader for long-token sequences.

Provides an IterableDataset that yields token-id sequences of requested context length
by slicing memory-mapped chunk files created by `scripts/preprocess_pg19.py`.
"""
from pathlib import Path
from typing import List, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class MemmapSequenceDataset(IterableDataset):
    """Iterable dataset that streams fixed-length sequences from memmap token files.

    Args:
        files: list of memmap file paths (int32 token ids)
        context_length: length of sequences to yield
        stride: step between sequences (defaults to context_length for non-overlapping)
    """

    def __init__(self, files: List[str], context_length: int, stride: int = None):
        super().__init__()
        self.files = [Path(p) for p in files]
        self.context_length = context_length
        self.stride = stride or context_length

    def _iter_single_file(self, path: Path) -> Iterator[torch.Tensor]:
        mm = np.memmap(path, dtype='int32', mode='r')
        n = mm.shape[0]
        i = 0
        while i + self.context_length <= n:
            seq = np.asarray(mm[i : i + self.context_length], dtype=np.int32)
            yield torch.from_numpy(seq).long()
            i += self.stride

    def __iter__(self) -> Iterator[torch.Tensor]:
        for p in self.files:
            yield from self._iter_single_file(p)


def list_memmap_files(directory: str, prefix: str = "pg19_") -> List[str]:
    p = Path(directory)
    files = sorted([str(f) for f in p.glob(f"{prefix}*_chunk_*.dat")])
    return files


if __name__ == "__main__":
    # small smoke test (won't run in CI) — prints first sequence shapes
    import sys

    if len(sys.argv) < 3:
        print("Usage: python src/data_loader.py <dir> <context_length>")
        raise SystemExit(1)
    files = list_memmap_files(sys.argv[1])
    ds = MemmapSequenceDataset(files, context_length=int(sys.argv[2]))
    it = iter(ds)
    x = next(it)
    print("Loaded sequence shape:", x.shape)
