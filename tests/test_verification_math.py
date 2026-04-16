"""Hand-rolled math spec for speculative-decoding verification.

Check 2 of the M3 post-analysis surfaced three defects in
`src/models/rasd_inference.py`:

  (#4) On rejection at position i, the bonus token is sampled from plain
       `p_target[i]` — Leviathan et al. require the *residual*
       `normalize(max(0, p_target[i] - p_draft[i]))`.

  (#5) After a round with n_accepted < k, the target's `past_key_values`
       still contains KV for all k+1 verify steps. The correct length is
       prior + n_accepted + 1 (n_accepted draft tokens + 1 bonus).

  (#6) The autoregressive verify loop feeds the target its *own argmax*
       into each next step rather than the draft's token, so
       `target_logits_v[:, i]` is conditioned on the wrong context for
       i > 0. This alone is enough to collapse acceptance on multi-token
       lookaheads.

This file is the spec the fix must satisfy. It is self-contained — the
reference `speculative_verify_round()` lives here (not in production) so
the tests can be written against a known-good implementation before we
touch `rasd_inference.py`. When the fix lands, we will point the
production function at the same semantics and the tests become the
regression guard.

Run:
    python -m pytest tests/test_verification_math.py -v
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reference implementation (the spec)
# ---------------------------------------------------------------------------

@torch.no_grad()
def speculative_verify_round_ref(
    cur_token: torch.Tensor,              # (B, 1)
    draft_seq: torch.Tensor,              # (B, k)
    draft_probs: torch.Tensor,            # (B, k, V)  — post-softmax
    target_forward: Callable,             # target_forward(input_ids, past_kv) -> (logits, new_past_kv)
    past_kv,
    temperature: float,
) -> Tuple[torch.Tensor, int, torch.Tensor, object, List[torch.Tensor]]:
    """Reference single-round speculative verify.

    Returns
        accepted     : (B, k) bool
        n_accepted   : int
        bonus_token  : (B, 1) — next-round cur_token
        new_past_kv  : truncated to prior + n_accepted + 1
        target_inputs_seen : list of input_ids the target was fed (spec #6 guard)
    """
    B, k = draft_seq.shape
    # --- Packed forward: target sees [cur_token, draft_seq[0..k-1]] at once ---
    t_input = torch.cat([cur_token, draft_seq], dim=1)     # (B, k+1)
    target_inputs_seen: List[torch.Tensor] = [t_input.clone()]
    t_logits, post_verify_kv = target_forward(t_input, past_kv)  # (B, k+1, V)

    if temperature == 0.0:
        tgt_argmax = t_logits[:, :k].argmax(dim=-1)
        accepted = (tgt_argmax == draft_seq)
    else:
        t_probs = F.softmax(t_logits / temperature, dim=-1)          # (B, k+1, V)
        idx     = draft_seq.unsqueeze(-1)                            # (B, k, 1)
        p_t     = t_probs[:, :k].gather(-1, idx).squeeze(-1)         # (B, k)
        p_d     = draft_probs.gather(-1, idx).squeeze(-1)            # (B, k)
        accept_prob = torch.clamp(p_t / (p_d + 1e-9), max=1.0)
        r = torch.rand_like(accept_prob)
        accepted = r < accept_prob

    first_reject = (accepted[0] == False).nonzero(as_tuple=False)
    n_acc = first_reject[0].item() if len(first_reject) > 0 else k

    # --- Bonus token: residual on reject, plain on full-accept ---
    if temperature == 0.0:
        if n_acc == k:
            bonus = t_logits[:, k].argmax(dim=-1, keepdim=True)
        else:
            bonus = t_logits[:, n_acc].argmax(dim=-1, keepdim=True)
    else:
        t_probs_full = F.softmax(t_logits / temperature, dim=-1)
        if n_acc == k:
            bonus_dist = t_probs_full[:, k]
        else:
            diff = torch.clamp(t_probs_full[:, n_acc] - draft_probs[:, n_acc], min=0.0)
            bonus_dist = diff / (diff.sum(-1, keepdim=True) + 1e-12)
        bonus = torch.multinomial(bonus_dist, num_samples=1)

    # --- Truncate KV to committed length: prior + n_acc + 1 ---
    prior_len = _kv_len(past_kv)
    new_past_kv = _kv_truncate(post_verify_kv, prior_len + n_acc + 1)

    return accepted, int(n_acc), bonus, new_past_kv, target_inputs_seen


# ---------------------------------------------------------------------------
# KV stub — a list of (k, v) tensors of shape (B, H, S, D)
# ---------------------------------------------------------------------------

def _make_kv(batch=1, heads=1, seq=0, dim=1, layers=1):
    return tuple(
        (torch.zeros(batch, heads, seq, dim), torch.zeros(batch, heads, seq, dim))
        for _ in range(layers)
    )


def _kv_len(past_kv) -> int:
    if past_kv is None:
        return 0
    return past_kv[0][0].shape[2]


def _kv_truncate(past_kv, new_len: int):
    return tuple((k[:, :, :new_len, :], v[:, :, :new_len, :]) for k, v in past_kv)


def _kv_extend(past_kv, added: int):
    """Append `added` zero positions to every layer."""
    out = []
    for k, v in past_kv:
        B, H, S, D = k.shape
        pad = torch.zeros(B, H, added, D)
        out.append((torch.cat([k, pad], dim=2), torch.cat([v, pad], dim=2)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Stub "target model" — logits depend on input_ids so we can detect mis-feeding
# ---------------------------------------------------------------------------

class FingerprintTarget:
    """Produces logits that encode which tokens it actually received.

    Logit at position p is a one-hot over vocab at input_ids[0, p]'s value,
    scaled so argmax returns input_ids[0, p] + 1 (mod V). This lets us tell,
    for each output position, what input token the model was fed — which
    directly exposes spec #6 violations.
    """

    def __init__(self, vocab: int):
        self.V = vocab
        self.calls: List[torch.Tensor] = []

    def __call__(self, input_ids, past_kv):
        self.calls.append(input_ids.clone())
        B, T = input_ids.shape
        logits = torch.full((B, T, self.V), -10.0)
        for b in range(B):
            for t in range(T):
                tok = int(input_ids[b, t].item())
                logits[b, t, (tok + 1) % self.V] = 10.0
        new_kv = _kv_extend(past_kv if past_kv is not None else _make_kv(), T)
        return logits, new_kv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

VOCAB = 8


def _probs(row):
    p = torch.tensor(row, dtype=torch.float32)
    return p / p.sum()


# ---- Spec #6: target must be fed DRAFT tokens, not its own argmax ----

def test_target_conditioned_on_draft_tokens():
    """The target's input at verify positions 1..k must equal draft_seq[0..k-1].

    A buggy autoregressive verify that feeds target.argmax() instead would
    produce input_ids where t_input[0, i+1] = argmax(target_logits[i]), not
    draft_seq[0, i]. We check that the reference passes the draft in directly.
    """
    torch.manual_seed(0)
    cur_token = torch.tensor([[3]])
    draft_seq = torch.tensor([[5, 2, 7, 1]])   # k=4
    draft_probs = torch.zeros(1, 4, VOCAB)
    draft_probs.scatter_(-1, draft_seq.unsqueeze(-1), 1.0)   # one-hot
    tgt = FingerprintTarget(VOCAB)

    _, _, _, _, inputs_seen = speculative_verify_round_ref(
        cur_token, draft_seq, draft_probs, tgt,
        past_kv=_make_kv(), temperature=0.0,
    )
    # The single packed forward must have seen [cur_token, draft_seq...]
    expected = torch.cat([cur_token, draft_seq], dim=1)
    assert torch.equal(inputs_seen[0], expected), (
        f"target was fed {inputs_seen[0].tolist()} but spec #6 requires "
        f"{expected.tolist()}"
    )


# ---- Spec #5: KV cache length = prior + n_accepted + 1 ----

def test_kv_truncated_to_commit_length():
    """After a round with n_acc rejections, new_past_kv length must equal
    prior + n_acc + 1. The buggy path keeps prior + k + 1."""
    torch.manual_seed(123)
    k = 5
    cur_token = torch.tensor([[0]])
    # Draft proposes low-target-prob tokens so we reject early
    draft_seq = torch.tensor([[2, 2, 2, 2, 2]])
    draft_probs = torch.full((1, k, VOCAB), 1e-6)
    draft_probs[:, :, 2] = 1.0 - 1e-6 * (VOCAB - 1)   # all mass on tok 2

    # Fingerprint target => argmax at position p is (input[p]+1) % V,
    # which is NOT 2 → greedy path rejects immediately (n_acc = 0)
    tgt = FingerprintTarget(VOCAB)
    prior_kv = _make_kv(seq=17)
    prior_len = _kv_len(prior_kv)

    _, n_acc, _, new_kv, _ = speculative_verify_round_ref(
        cur_token, draft_seq, draft_probs, tgt, prior_kv, temperature=0.0,
    )
    assert n_acc == 0, f"expected immediate rejection, got n_acc={n_acc}"
    assert _kv_len(new_kv) == prior_len + n_acc + 1, (
        f"KV length = {_kv_len(new_kv)}, expected {prior_len + n_acc + 1} "
        f"(prior={prior_len}, n_acc={n_acc}). Spec #5 violated."
    )


def test_kv_length_on_full_acceptance():
    """When all k draft tokens accept, KV must extend by k+1."""
    torch.manual_seed(0)
    k = 3
    cur_token = torch.tensor([[0]])
    # Target's argmax at pos p is (input[p]+1) % V.
    # We want draft_seq[i] == target_argmax_at_pos_i_when_fed_draft_correctly.
    # Packed feed: [cur_token=0, d0, d1, d2] → argmax at 0 = 1 (so d0 must be 1)
    # argmax at 1 (input d0=1) = 2 (so d1 must be 2); argmax at 2 (d1=2) = 3.
    draft_seq = torch.tensor([[1, 2, 3]])
    draft_probs = torch.zeros(1, k, VOCAB)
    draft_probs.scatter_(-1, draft_seq.unsqueeze(-1), 1.0)
    tgt = FingerprintTarget(VOCAB)
    prior_kv = _make_kv(seq=4)
    prior_len = _kv_len(prior_kv)

    _, n_acc, _, new_kv, _ = speculative_verify_round_ref(
        cur_token, draft_seq, draft_probs, tgt, prior_kv, temperature=0.0,
    )
    assert n_acc == k
    assert _kv_len(new_kv) == prior_len + k + 1


# ---- Spec #4: residual resample distribution on rejection ----

def test_residual_resample_on_rejection():
    """Empirical: when draft[0] is rejected, bonus must follow
    normalize(max(0, p_t - p_d)), not raw p_t.

    Construction:
      p_draft[0] puts 0.9 on token 0, ~0.014 on tokens 1..7  (V=8)
      p_target[0] puts 0.1 on token 0, 0.8 on token 1, ~0.017 on 2..7
      p_target[1] (bonus slot after reject at pos 0): 0.1 on tok0, 0.8 on tok1, rest ~0.017
      residual = max(0, p_t - p_d) normalized
               ≈ zero on tok 0 (0.1 - 0.9 clipped to 0)
               ≈ concentrated on tok 1 (0.8 - 0.014 ≈ 0.786 → normalize)

    Test:
      1. Draft always proposes tok 0 (p_d[0]=0.9 there).
      2. Accept prob = p_t[0] / p_d[0] = 0.1/0.9 ≈ 0.111 → reject ~89% of trials.
      3. Among rejected trials, bonus should come from residual distribution
         (≈ 0 prob of tok 0, ≈ high prob of tok 1).
      4. Buggy code would sample bonus from p_t → tok 1 ~80% but tok 0 ~10%.
    """
    torch.manual_seed(2026)
    k = 1
    V = 8

    p_draft_row = torch.full((V,), 0.014)
    p_draft_row[0] = 0.9
    p_draft_row = p_draft_row / p_draft_row.sum()

    p_tgt_row = torch.full((V,), 0.017)
    p_tgt_row[0] = 0.1
    p_tgt_row[1] = 0.8
    p_tgt_row = p_tgt_row / p_tgt_row.sum()

    # Stub target that returns fixed softmax probs as logits (log of probs so
    # softmax(logits) = probs exactly).
    class FixedTarget:
        def __call__(self, input_ids, past_kv):
            B, T = input_ids.shape
            logits = torch.log(p_tgt_row.clamp(min=1e-9)).view(1, 1, V).expand(B, T, V).clone()
            new_kv = _kv_extend(past_kv if past_kv is not None else _make_kv(), T)
            return logits, new_kv

    cur_token = torch.tensor([[4]])
    draft_seq = torch.tensor([[0]])
    draft_probs = p_draft_row.view(1, 1, V).clone()
    tgt = FixedTarget()

    n_trials = 4000
    reject_bonus_counts = torch.zeros(V)
    n_rejections = 0
    for _ in range(n_trials):
        accepted, n_acc, bonus, _, _ = speculative_verify_round_ref(
            cur_token, draft_seq, draft_probs, tgt,
            past_kv=_make_kv(), temperature=1.0,
        )
        if n_acc == 0:   # rejection at position 0
            reject_bonus_counts[int(bonus.item())] += 1
            n_rejections += 1

    assert n_rejections > n_trials * 0.7, (
        f"expected ~89% rejection rate, got {n_rejections}/{n_trials}"
    )
    # residual: tok 0 is forbidden (p_t - p_d < 0 → clipped to 0)
    p_tok0 = reject_bonus_counts[0].item() / n_rejections
    assert p_tok0 < 0.03, (
        f"Spec #4: tok 0 has residual mass 0 but sampled with p={p_tok0:.3f}. "
        f"This means bonus is being drawn from raw p_target, not residual."
    )
    # residual heavily favors tok 1
    p_tok1 = reject_bonus_counts[1].item() / n_rejections
    assert p_tok1 > 0.9, (
        f"Spec #4: residual concentrates on tok 1 (≈0.97) but got p={p_tok1:.3f}."
    )


# ---- Sanity: full acceptance bonus comes from raw p_target[k] ----

def test_bonus_on_full_acceptance_is_plain_target():
    torch.manual_seed(7)
    k = 2
    V = 4
    # Target argmax pattern: pos p -> (input[p]+1) % 4
    # cur_token=0 → arg at pos 0 = 1; feed d0=1 → arg at pos 1 = 2; feed d1=2 → arg at pos 2 (bonus) = 3.
    cur_token = torch.tensor([[0]])
    draft_seq = torch.tensor([[1, 2]])
    draft_probs = torch.zeros(1, k, V)
    draft_probs.scatter_(-1, draft_seq.unsqueeze(-1), 1.0)
    tgt = FingerprintTarget(V)

    _, n_acc, bonus, _, _ = speculative_verify_round_ref(
        cur_token, draft_seq, draft_probs, tgt,
        past_kv=_make_kv(), temperature=0.0,
    )
    assert n_acc == k
    # Bonus under greedy = argmax of t_logits[:, k]  — fed d1=2 → arg = 3.
    assert int(bonus.item()) == 3, (
        f"full-acceptance bonus = argmax target_logits[:, k] = 3, got {bonus.item()}"
    )


# ---- Acceptance mask math (locks in the existing helper) ----

def test_acceptance_ratio_math():
    """For p_t = 0.2, p_d = 0.8 → accept with prob 0.25 (empirical)."""
    torch.manual_seed(0)
    V = 4
    p_t = torch.tensor([0.2, 0.6, 0.1, 0.1])
    p_d = torch.tensor([0.8, 0.1, 0.05, 0.05])
    accept_prob = torch.clamp(p_t[0] / (p_d[0] + 1e-9), max=1.0)
    assert abs(accept_prob.item() - 0.25) < 1e-4

    # Empirical
    n = 10_000
    accepts = (torch.rand(n) < accept_prob).float().mean().item()
    assert abs(accepts - 0.25) < 0.02, f"empirical accept rate {accepts}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
