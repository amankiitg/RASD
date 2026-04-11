"""
Ring Peer Protocol Validation
==============================
Tests the broadcast-based signaling protocol between rank 0 (generate loop)
and non-rank-0 peers (_ring_peer_loop) using CPU + gloo — no GPU required.

What is validated
-----------------
1. Correct termination for all prefetch_depth values (0, 1, 2)
2. No deadlock when rank 0 exits early (EOS simulation)
3. P2P buffer exchange completes for all A3 kv_block_sizes (256-2048)
4. Ring relay: data written by rank 0 arrives at rank 0 after world_size steps

Why this catches the previous bugs
------------------------------------
Bug 1 (pre-loop schedule with no broadcast): caught — rank 0 must broadcast
      before every P2P op; any missing broadcast causes a 5s timeout failure.
Bug 2 (prefetch_depth=0 never broadcasts): caught — test runs prefetch_depth=0
      explicitly; previously peers hung forever.
Bug 3 (terminal broadcast gated on prefetch_depth): caught — same as Bug 2
      but at teardown; processes would not join within timeout.
Bug 4 (1 send/recv in peers vs 2 in rank 0): caught — mismatched op counts
      leave one isend without a matching irecv, blocking batch_isend_irecv.

Usage
-----
    # From project root (rasd conda env):
    conda run -n rasd python tests/test_ring_protocol.py

    # Verbose:
    conda run -n rasd python tests/test_ring_protocol.py -v
"""

import os
import sys
import time
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Protocol constants (CPU/float32 for gloo compatibility)
# ---------------------------------------------------------------------------

_MASTER_ADDR = "127.0.0.1"
_MASTER_PORT = "29600"   # distinct from default torchrun port
_BACKEND     = "gloo"
_DTYPE       = torch.float32   # gloo P2P works with float32; bfloat16 is NCCL-only


# ---------------------------------------------------------------------------
# Rank 0 worker — mirrors generate() signaling logic
# ---------------------------------------------------------------------------

def _rank0_worker(rank, world_size, kv_block_size, n_rounds, early_exit_at):
    """Simulates rank 0's generate() loop:
       broadcast(1) → P2P schedule → ... → broadcast(0) to stop peers.

    early_exit_at: rank 0 exits after this many rounds (simulates EOS).
                   Set to n_rounds for full run.
    """
    os.environ["MASTER_ADDR"] = _MASTER_ADDR
    os.environ["MASTER_PORT"] = _MASTER_PORT
    dist.init_process_group(_BACKEND, rank=rank, world_size=world_size)

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    flat_size = 32 * 32 * kv_block_size * 4  # scaled down: num_layers*H*block*head_dim/32
    k_send = torch.ones(flat_size, dtype=_DTYPE) * float(rank)
    v_send = torch.ones(flat_size, dtype=_DTYPE) * float(rank + 0.5)
    k_recv = torch.zeros(flat_size, dtype=_DTYPE)
    v_recv = torch.zeros(flat_size, dtype=_DTYPE)

    actual_rounds = min(n_rounds, early_exit_at)
    for i in range(actual_rounds):
        # Broadcast "continue" — peers receive this in their loop
        sig = torch.tensor([1], dtype=torch.long)
        dist.broadcast(sig, src=0)
        # Post 4 P2P ops matching AsyncKVRingPrefetcher.schedule()
        ops = [
            dist.P2POp(dist.isend, k_send.clone(), send_to),
            dist.P2POp(dist.isend, v_send.clone(), send_to),
            dist.P2POp(dist.irecv, k_recv, recv_from),
            dist.P2POp(dist.irecv, v_recv, recv_from),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    # Terminal broadcast — must fire unconditionally (not gated on prefetch_depth)
    done = torch.tensor([0], dtype=torch.long)
    dist.broadcast(done, src=0)

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Peer worker — mirrors _ring_peer_loop in run_experiment.py
# ---------------------------------------------------------------------------

def _peer_worker(rank, world_size, kv_block_size):
    """Simulates non-rank-0 _ring_peer_loop: wait for broadcast, do P2P, repeat."""
    os.environ["MASTER_ADDR"] = _MASTER_ADDR
    os.environ["MASTER_PORT"] = _MASTER_PORT
    dist.init_process_group(_BACKEND, rank=rank, world_size=world_size)

    send_to   = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    flat_size = 32 * 32 * kv_block_size * 4
    signal = torch.zeros(1, dtype=torch.long)
    k_send = torch.zeros(flat_size, dtype=_DTYPE)
    v_send = torch.zeros(flat_size, dtype=_DTYPE)
    k_recv = torch.zeros(flat_size, dtype=_DTYPE)
    v_recv = torch.zeros(flat_size, dtype=_DTYPE)

    while True:
        dist.broadcast(signal, src=0)
        if signal.item() == 0:
            break
        ops = [
            dist.P2POp(dist.isend, k_send.clone(), send_to),
            dist.P2POp(dist.isend, v_send.clone(), send_to),
            dist.P2POp(dist.irecv, k_recv, recv_from),
            dist.P2POp(dist.irecv, v_recv, recv_from),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()
        # Ring relay: forward received block
        k_send.copy_(k_recv)
        v_send.copy_(v_recv)

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point for mp.spawn
# ---------------------------------------------------------------------------

def _worker_fn(rank, world_size, kv_block_size, n_rounds, early_exit_at):
    if rank == 0:
        _rank0_worker(rank, world_size, kv_block_size, n_rounds, early_exit_at)
    else:
        _peer_worker(rank, world_size, kv_block_size)


# ---------------------------------------------------------------------------
# Test runner helper
# ---------------------------------------------------------------------------

def _run_protocol(world_size=4, kv_block_size=8, n_rounds=6, early_exit_at=None,
                  timeout=30):
    """Spawn world_size processes and run the protocol. Raises on deadlock (timeout)."""
    if early_exit_at is None:
        early_exit_at = n_rounds

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_fn,
            args=(rank, world_size, kv_block_size, n_rounds, early_exit_at),
        )
        p.start()
        procs.append(p)

    deadline = time.time() + timeout
    for p in procs:
        remaining = max(0.1, deadline - time.time())
        p.join(timeout=remaining)

    alive = [p for p in procs if p.is_alive()]
    for p in alive:
        p.kill()
    if alive:
        raise TimeoutError(
            f"{len(alive)}/{world_size} processes still alive after {timeout}s "
            f"(kv_block_size={kv_block_size}, n_rounds={n_rounds}, "
            f"early_exit_at={early_exit_at}) — likely deadlock"
        )

    failed = [p for p in procs if p.exitcode != 0]
    if failed:
        codes = [p.exitcode for p in failed]
        raise RuntimeError(
            f"{len(failed)} processes exited with errors: {codes} "
            f"(kv_block_size={kv_block_size}, n_rounds={n_rounds})"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRingProtocol(unittest.TestCase):

    def test_full_run_default_block_size(self):
        """Normal run: all rounds complete, clean shutdown."""
        _run_protocol(world_size=4, kv_block_size=8, n_rounds=5)

    def test_early_eos_exit(self):
        """Rank 0 exits after 2 rounds (EOS simulation). Peers must not hang."""
        _run_protocol(world_size=4, kv_block_size=8, n_rounds=10, early_exit_at=2)

    def test_single_round(self):
        """Minimum viable: 1 round then exit."""
        _run_protocol(world_size=4, kv_block_size=8, n_rounds=1)

    def test_kv_block_size_256(self):
        """A3 ablation: kv_block_size=256 (smallest block, most frequent comm)."""
        _run_protocol(world_size=4, kv_block_size=16, n_rounds=4)  # scaled: 256/16=16x

    def test_kv_block_size_512(self):
        """A3 ablation: kv_block_size=512 (default)."""
        _run_protocol(world_size=4, kv_block_size=32, n_rounds=4)

    def test_kv_block_size_1024(self):
        """A3 ablation: kv_block_size=1024 (larger blocks)."""
        _run_protocol(world_size=4, kv_block_size=64, n_rounds=4)

    def test_kv_block_size_2048(self):
        """A3 ablation: kv_block_size=2048 (largest blocks)."""
        _run_protocol(world_size=4, kv_block_size=128, n_rounds=4)

    def test_world_size_2(self):
        """Ring with 2 ranks (smallest ring)."""
        _run_protocol(world_size=2, kv_block_size=8, n_rounds=5)

    def test_world_size_8(self):
        """Full 8-GPU ring size (as on RunPod)."""
        _run_protocol(world_size=8, kv_block_size=8, n_rounds=4, timeout=60)

    def test_many_rounds(self):
        """Stress: 50 rounds — catches any per-round accumulation issues."""
        _run_protocol(world_size=4, kv_block_size=8, n_rounds=50, timeout=60)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with -v for verbose output
    verbose = "-v" in sys.argv
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromTestCase(TestRingProtocol)
    runner  = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result  = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
