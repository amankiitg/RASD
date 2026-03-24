import sys
import platform
import torch
import transformers
import datasets
import wandb
import torch.distributed as dist

print("=" * 50)
print("  RASD Environment Check")
print("=" * 50)

# System
print(f"\n── System ──────────────────────────────")
print(f"OS:             {platform.system()} {platform.mac_ver()[0]}")
print(f"Python:         {sys.version.split()[0]}")
print(f"Architecture:   {platform.machine()}")

# PyTorch + backend
print(f"\n── PyTorch ─────────────────────────────")
print(f"PyTorch:        {torch.__version__}")

# CUDA (won't be available on Mac)
print(f"CUDA available: {torch.cuda.is_available()}")

# MPS — Apple Silicon GPU
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS available:  {mps_available}")

if mps_available:
    device = torch.device("mps")
    print(f"Active backend: MPS (Apple Silicon GPU) ✓")
    # Quick tensor smoke test on MPS
    try:
        x = torch.ones(3, 3).to(device)
        y = x * 2
        assert y.sum().item() == 18.0
        print(f"MPS tensor test: passed ✓")
    except Exception as e:
        print(f"MPS tensor test: FAILED — {e}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Active backend: CUDA")
    print(f"CUDA version:   {torch.version.cuda}")
    print(f"GPU count:      {torch.cuda.device_count()}")
    print(f"GPU name:       {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print(f"Active backend: CPU only ⚠️  (no MPS or CUDA)")

# ML libraries
print(f"\n── ML Libraries ────────────────────────")
print(f"Transformers:   {transformers.__version__}")
print(f"Datasets:       {datasets.__version__}")
print(f"WandB:          {wandb.__version__}")

# Distributed (CPU-based on Mac — still useful for logic testing)
print(f"\n── Distributed ─────────────────────────")
print(f"dist available: {dist.is_available()}")
if dist.is_available():
    print(f"Backends:       gloo={dist.is_gloo_available()}  "
          f"mpi={dist.is_mpi_available()}  "
          f"nccl={dist.is_nccl_available()}")
    print(f"Note: nccl=False is expected on Mac (NVIDIA only)")

# Quick model smoke test
print(f"\n── Smoke Test ──────────────────────────")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tokens = tok("Ring Attention is awesome", return_tensors="pt")
    input_ids = tokens["input_ids"].to(device)
    print(f"Tokenizer test: passed ✓  ({input_ids.shape[1]} tokens on {device})")
except Exception as e:
    print(f"Tokenizer test: FAILED — {e}")

# Summary
print(f"\n── Summary ─────────────────────────────")
if mps_available:
    print("✅ Ready for local Mac development (MPS backend)")
    print("⚠️  For full 8-GPU experiments use environment_gpu.yml on Linux cluster")
elif torch.cuda.is_available():
    print("✅ Ready for GPU experiments")
else:
    print("⚠️  CPU only — fine for code testing, slow for actual experiments")

print("=" * 50)