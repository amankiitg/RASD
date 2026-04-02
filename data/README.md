# Data: PG-19 Preprocessing

## Dataset

[PG-19](https://huggingface.co/datasets/pg19) is a long-context language modelling benchmark built from Project Gutenberg books published before 1919. It contains ~28,000 books (~2.9B tokens) split into train/validation/test sets.

## Preprocessing Pipeline

The script `scripts/preprocess_pg19.py` downloads, tokenizes, and chunks PG-19 into memory-mapped files suitable for training on million-token sequences.

### Steps

1. **Download** — PG-19 is streamed directly from HuggingFace `datasets` (no manual download needed).
2. **Tokenize** — Each book is tokenized with the GPT-2 tokenizer (`gpt2`) using `AutoTokenizer` from `transformers`. Special tokens are not added between documents.
3. **Concatenate & Chunk** — Token streams from successive books are concatenated into a single long stream, then sliced into fixed-size chunks (default: 1,000,000 tokens each) with no overlap.
4. **Write memmap files** — Each chunk is written as a raw `int32` NumPy memmap file (`*.dat`) for O(1) random access without loading the full dataset into RAM.
5. **Metadata** — A `pg19_<split>_metadata.json` file records each chunk's filename and token count.

### Running the Script

```bash
# Default: train split, GPT-2 tokenizer, 1M-token chunks
python scripts/preprocess_pg19.py

# Custom options
python scripts/preprocess_pg19.py \
    --split validation \
    --tokenizer gpt2 \
    --out data/processed/pg19 \
    --chunk-size 1000000
```

Output files go to `data/processed/pg19/` by default:

```
data/processed/pg19/
  pg19_train_chunk_0.dat       # 1M int32 token ids
  pg19_train_chunk_1.dat
  ...
  pg19_train_metadata.json     # {"chunks": [{"file": "...", "length": 1000000}, ...]}
```

### Using the Data Loader

```python
from src.data_loader import MemmapSequenceDataset, list_memmap_files

files = list_memmap_files("data/processed/pg19", prefix="pg19_train")
dataset = MemmapSequenceDataset(files, context_length=131072)  # 128k tokens

for batch in dataset:
    # batch: torch.LongTensor of shape (131072,)
    ...
```

The dataset is an `IterableDataset` — it reads directly from memmap files without loading everything into RAM, making it safe for million-token sequences on machines with limited CPU memory.

## Directory Layout

```
data/
  raw/          # (empty — streaming download, nothing stored here)
  processed/
    pg19/
      pg19_train_chunk_*.dat
      pg19_train_metadata.json
      pg19_validation_chunk_*.dat
      pg19_validation_metadata.json
  README.md     # this file
```

## Storage Requirements

| Split      | Books  | Approx Tokens | Approx Disk (int32) |
|------------|--------|---------------|----------------------|
| train      | 28,602 | ~2.87B        | ~11 GB               |
| validation | 50     | ~4.9M         | ~19 MB               |
| test       | 100    | ~9.9M         | ~38 MB               |

## Notes

- The tokenizer must match across preprocessing and training. Default is `gpt2` (vocab size 50,257).
- Chunks at the end of each split may be smaller than `chunk_size` (the remainder is kept).
- For benchmarking purposes, the `benchmark_baselines.py` script uses synthetic `torch.randn` inputs — you do **not** need to run preprocessing before running benchmarks.
