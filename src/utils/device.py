"""
Device selector for RASD — auto-detects MPS (MacBook), CUDA (RunPod), or CPU.

Capabilities by backend
-----------------------
                    CUDA (RunPod)   MPS (MacBook)   CPU
  LLM inference         yes             yes         yes (slow)
  4-bit quantization    yes             NO          NO
  torch.distributed     yes (nccl)      NO          yes (gloo, slow)
  ring P2P comm         yes             NO          NO
  CUDA streams          yes             NO          NO

Usage
-----
    from src.utils.device import get_device, DeviceCapabilities

    device = get_device()                        # torch.device
    caps   = DeviceCapabilities.detect()

    # Load a model without quantization on MPS/CPU:
    if caps.supports_quantization:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, ...)
    model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_cfg)

    # device_map for from_pretrained:
    model = AutoModelForCausalLM.from_pretrained(..., **caps.hf_device_map_kwargs())
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DeviceCapabilities:
    device: torch.device
    device_type: str          # "cuda" | "mps" | "cpu"
    supports_quantization: bool    # bitsandbytes 4-bit NF4
    supports_distributed: bool     # torch.distributed NCCL ring comm
    supports_cuda_streams: bool    # torch.cuda.Stream / Event

    @classmethod
    def detect(cls, local_rank: int = 0) -> "DeviceCapabilities":
        """Auto-detect capabilities. local_rank used for multi-GPU CUDA only."""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            logger.info("Device: %s (CUDA — RunPod mode, all features enabled)", device)
            return cls(
                device=device,
                device_type="cuda",
                supports_quantization=True,
                supports_distributed=True,
                supports_cuda_streams=True,
            )
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(
                "Device: mps (MacBook M-series — inference only, "
                "quantization/distributed/ring-comm disabled)"
            )
            return cls(
                device=device,
                device_type="mps",
                supports_quantization=False,
                supports_distributed=False,
                supports_cuda_streams=False,
            )
        logger.warning("Device: cpu (no GPU found — inference will be slow)")
        return cls(
            device=torch.device("cpu"),
            device_type="cpu",
            supports_quantization=False,
            supports_distributed=False,
            supports_cuda_streams=False,
        )

    def hf_device_map_kwargs(self) -> dict:
        """Kwargs to pass to AutoModelForCausalLM.from_pretrained for device placement.

        CUDA:  device_map={"": local_rank}  — places all layers on one GPU
        MPS:   no device_map (transformers handles MPS natively via .to())
        CPU:   no device_map
        """
        if self.device_type == "cuda":
            rank = self.device.index or 0
            return {"device_map": {"": rank}}
        # MPS and CPU: let transformers place the model, then we call .to(device)
        return {}

    def peak_memory_mb(self) -> float:
        """Return peak allocated memory in MB (CUDA only; 0 on MPS/CPU)."""
        if self.device_type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
        return 0.0

    def reset_peak_memory(self):
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self):
        """Synchronize the device (noop on CPU)."""
        if self.device_type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device_type == "mps":
            torch.mps.synchronize()
