"""Tests for how the teacher-forced probe is INVOKED, as opposed to what it
measures.

The probe's arithmetic is covered by test_teacher_forced_losslessness.py and
test_mlsys_cap_smoke_check.py. What failed on 2026-10-09T00:00Z was the wiring,
and the failure mode is worth stating precisely because it does not look like a
wiring bug at the time: `gen_ids` is populated on rank 0 only, the entry guard
tested it, so rank 0 entered a ring collective alone, completed 7036 of them by
itself, and then sat in the NCCL watchdog for 3600 s. Nothing raised, nothing
logged, and the campaign burned an hour of 8xA100-80GB before reporting
anything.

So these tests are about three properties, each of which the one-rank harness
could never have caught -- it caught none of them, and that is why the probe
shipped broken:

  1. every rank enters, decided by the run flag and NOT by rank-local state;
  2. every rank judges the SAME stream (the ids are broadcast, not assumed);
  3. one rank's failure is every rank's failure, so no rank is left waiting in
     a collective the failing rank no longer enters.
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import run_experiment  # noqa: E402

RUN_EXP_SRC = (REPO_ROOT / "run_experiment.py").read_text()


class _FakeEngine:
    """Records what it was asked to do; no torch, no ring."""

    def __init__(self, result=None, raises=None):
        self.calls = []
        self.result = result if result is not None else {"max_shortfall": 0.375}
        self.raises = raises

    def teacher_forced_probe(self, prompt_ids, tokens, budget_s=None):
        self.calls.append({"prompt_ids": prompt_ids, "tokens": tokens,
                           "budget_s": budget_s})
        if self.raises is not None:
            raise self.raises
        return dict(self.result)


class _FakeDist:
    """A ring that shares state the way the real one does: whatever rank 0 puts
    in `broadcast_object_list`'s box is what every rank sees, and
    `all_gather_object` sees every rank's value."""

    def __init__(self, ranks):
        self.ranks = ranks          # list of dicts, one per rank

    def broadcast_object_list(self, box, src=0):
        box[0] = self.ranks[0]["outbox"][0]
        box[1] = self.ranks[0]["outbox"][1]

    def all_gather_object(self, out, value):
        for i in range(len(out)):
            out[i] = self.ranks[i]["value"]

    def all_reduce(self, tensor, op=None):
        pass


def _run_all_ranks(monkeypatch, world_size, rank_payloads, engine_factory=None,
                   dist_fake=None):
    """Drive `run_teacher_forced_probe` once per rank and return the verdicts.

    The real function reads `dist` from inside itself, so the fake is installed
    into `torch.distributed` for the duration. Ranks share the fake `dist`, which
    is what lets the broadcast and the agreement be observed from the outside.
    """
    import torch

    verdicts = []
    for local_rank in range(world_size):
        gen_ids, engine_input_ids, engine = rank_payloads[local_rank]
        if dist_fake is not None:
            monkeypatch.setattr(torch, "distributed", dist_fake, raising=False)
        verdicts.append(
            run_experiment.run_teacher_forced_probe(
                engine, {"spec_steps": 4, "context_length": 131072},
                gen_ids, engine_input_ids,
                world_size=world_size, local_rank=local_rank))
    return verdicts


class TestEveryRankEnters:
    def test_a_rank_without_gen_ids_still_enters_the_probe(self, monkeypatch):
        """THE REGRESSION. On ranks 1-7 `gen_ids` is None by construction (the
        metric guards in RASDInference.generate are `self._rank == 0`). The old
        entry condition was `and gen_ids and engine_input_ids`, so those ranks
        skipped the probe while rank 0 ran it -- and the probe is a ring
        collective, so rank 0 waited for peers that were never coming."""
        import torch

        engines = [_FakeEngine() for _ in range(2)]
        shared = _FakeDist([{"outbox": ([7, 8, 9], [1, 7, 8, 9]), "value": ""},
                            {"outbox": (None, None), "value": ""}])
        payloads = [(  # rank 0: the only rank with the ids
            [7, 8, 9], [1, 7, 8, 9], engines[0]),
            (None, None, engines[1])]  # rank 1: None, exactly as in production
        verdicts = _run_all_ranks(monkeypatch, 2, payloads, dist_fake=shared)
        assert engines[0].calls, "rank 0 must probe"
        assert engines[1].calls, (
            "rank 1 must ENTER the probe: skipping it leaves rank 0 inside a "
            "ring collective alone, which is the 2026-10-09T00:00Z deadlock")

    def test_every_rank_judges_the_same_stream(self, monkeypatch):
        import torch

        engines = [_FakeEngine(), _FakeEngine()]
        shared = _FakeDist([{"outbox": ([7, 8, 9], [1, 7, 8, 9]), "value": ""},
                            {"outbox": (None, None), "value": ""}])
        payloads = [([7, 8, 9], [1, 7, 8, 9], engines[0]),
                    (None, None, engines[1])]
        _run_all_ranks(monkeypatch, 2, payloads, dist_fake=shared)
        assert engines[1].calls[0]["tokens"] == [7, 8, 9], (
            "rank 1 was handed rank-local ids instead of the broadcast ones; "
            "each rank must judge the stream the run actually emitted")
        assert engines[1].calls[0]["prompt_ids"] == [1, 7, 8, 9]

    def test_the_entry_condition_is_the_run_flag(self):
        """A source guard, because the bug was a guard. `and gen_ids` must not
        come back: it reads as a cheap safety check and is a ring deadlock."""
        body = inspect.getsource(run_experiment._run_single_worker)
        entry = body[body.index("teacher_forced_check"):]
        entry = entry[:entry.index("\n") + 1]
        assert "gen_ids" not in entry, (
            "the probe's entry condition must be the run flag alone; a "
            "rank-local id check skips the probe on ranks 1-7")


class TestRanksAgreeOnFailure:
    def test_one_rank_s_error_fails_every_rank(self, monkeypatch):
        import torch

        engines = [_FakeEngine(), _FakeEngine(raises=RuntimeError("rank 1 died"))]
        shared = _FakeDist([{"outbox": ([7, 8], [1, 7, 8]), "value": ""},
                            {"outbox": (None, None), "value": ""}])
        payloads = [([7, 8], [1, 7, 8], engines[0]), (None, None, engines[1])]
        # The agreement collective is what carries the messages; feed each rank
        # its own outcome the way all_gather_object would.
        def gather(out, value):
            out[0] = "" if engines[0].raises is None else ""
            out[1] = "RuntimeError: rank 1 died"
        shared.all_gather_object = gather
        verdicts = _run_all_ranks(monkeypatch, 2, payloads, dist_fake=shared)
        assert all(v.get("error") for v in verdicts), (
            "a rank that fails alone strands its peers; every rank must record "
            "the failure")
        assert "rank 1 died" in verdicts[0]["error"], (
            "rank 0 must record the FAILING rank's message, not a generic one")

    def test_all_ranks_succeeding_records_no_error(self, monkeypatch):
        import torch

        engines = [_FakeEngine(), _FakeEngine()]
        shared = _FakeDist([{"outbox": ([7, 8], [1, 7, 8]), "value": ""},
                            {"outbox": (None, None), "value": ""}])
        payloads = [([7, 8], [1, 7, 8], engines[0]), (None, None, engines[1])]
        verdicts = _run_all_ranks(monkeypatch, 2, payloads, dist_fake=shared)
        assert not any(v.get("error") for v in verdicts)
        assert verdicts[1]["max_shortfall"] == 0.375, (
            "the agreement must not discard a good measurement")

    def test_a_failed_broadcast_is_reported_not_guessed(self, monkeypatch):
        """If the ids cannot be shared, each rank must NOT fall back to its own
        (empty) state and run a probe on a different stream."""
        import torch

        class _BrokenBroadcast(_FakeDist):
            def broadcast_object_list(self, box, src=0):
                raise RuntimeError("pg is gone")

        engines = [_FakeEngine(), _FakeEngine()]
        shared = _BrokenBroadcast([{"outbox": (None, None), "value": ""},
                                   {"outbox": (None, None), "value": ""}])
        payloads = [([7, 8], [1, 7, 8], engines[0]), (None, None, engines[1])]
        verdicts = _run_all_ranks(monkeypatch, 2, payloads, dist_fake=shared)
        assert all("broadcast failed" in v["error"] for v in verdicts)
        assert not engines[0].calls and not engines[1].calls, (
            "neither rank may run a probe whose input could not be agreed")


class TestBudget:
    def test_the_budget_is_passed_to_the_probe(self, monkeypatch):
        engine = _FakeEngine()
        run_experiment.run_teacher_forced_probe(
            engine, {"spec_steps": 4}, [7, 8], [1, 7, 8],
            world_size=1, local_rank=0)
        assert engine.calls[0]["budget_s"] == run_experiment.PROBE_BUDGET_S
        assert run_experiment.PROBE_BUDGET_S > 0, (
            "a disabled budget is the hang this was added to prevent")

    def test_the_process_group_watchdog_is_bounded_by_the_budget(self):
        """The in-process check cannot fire on a rank blocked INSIDE a
        collective -- it never gets a turn. The watchdog is the only thing that
        can end that rank, and it was an hour."""
        assert run_experiment.PROBE_PG_TIMEOUT_MIN <= 60, (
            "the process group timeout is the outer bound on a deadlock")
        src = RUN_EXP_SRC
        assert "timeout=timedelta(hours=1)" not in src, (
            "a one-hour NCCL watchdog is how the 2026-10-09 deadlock cost 3600s")

    def test_an_over_budget_probe_raises_rather_than_hanging(self):
        from src.models import rasd_inference as ri

        class _Engine:
            _world_size = 1
            target_model = None

            def __init__(self):
                import torch
                self.target_model = torch.nn.Linear(1, 1)

        eng = _Engine()
        check = ri.RASDInference._check_probe_budget.__get__(eng)
        import time as _time
        # inside the budget: silent
        check(_time.monotonic(), 10_000.0, 3)
        # past it: a typed error, not a log line and not a wait
        with pytest.raises(ri.ProbeBudgetExceeded) as exc:
            check(_time.monotonic() - 10.0, 0.0001, 3)
        assert "budget" in str(exc.value)

    def test_a_zero_budget_disables_the_check(self):
        import torch

        from src.models import rasd_inference as ri
        eng = ri.RASDInference.__new__(ri.RASDInference)
        eng.target_model = torch.nn.Linear(1, 1)
        eng._world_size = 1
        ri.RASDInference._check_probe_budget(eng, 0.0, 0.0, 1)  # must not raise

    def test_the_probe_records_the_budget_it_ran_under(self):
        """Without this the sidecar cannot distinguish "fast" from "about to
        time out", and the budget could be raised only after a failure."""
        src = inspect.getsource(
            __import__("src.models.rasd_inference", fromlist=["x"])
            .RASDInference.teacher_forced_probe)
        assert '"budget_s": budget_s' in src
        assert '"elapsed_s"' in src


class TestOnlyTheGatedPairIsProbed:
    def test_the_probe_runs_on_the_bf16_pair_only(self):
        """Probing the NF4 rows bought two figures that could never gate (NF4
        noise flips 20% of argmaxes at 128k, so no threshold separates it from a
        real defect) at the cost of the probe's wall clock on two more rows."""
        import yaml

        cfg = yaml.safe_load(
            (REPO_ROOT / "configs/mlsys_engine_cap_smoke.yml").read_text())
        levels = cfg["CAP_SMOKE"]["levels"]
        probed = [lv["id"] for lv in levels if lv.get("teacher_forced_check")]
        assert probed == ["CAPS_prefix64_bf16", "CAPS_prefix64_bf16_targetonly"], (
            f"only the bf16 pair may be probed, got {probed}")
        for lv in levels:
            if lv["id"] in probed:
                assert lv.get("kv_quant") is False, (
                    "the probed pair must be the bf16 one, not NF4")

    def test_the_gated_pair_is_still_present_for_the_checker(self):
        """Removing probes must not remove the pair the CHECKER gates on: with
        no pair measuring a GATED_KV dtype the anti-vanishing guard in
        mlsys_cap_smoke_check.py would (correctly) fail the stage."""
        import yaml

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import mlsys_cap_smoke_check as check

        cfg = yaml.safe_load(
            (REPO_ROOT / "configs/mlsys_engine_cap_smoke.yml").read_text())
        levels = cfg["CAP_SMOKE"]["levels"]
        gated = [lv["id"] for lv in levels
                 if lv.get("teacher_forced_check") and lv.get("kv_quant") is False]
        assert gated, "no probed pair would measure a GATED_KV dtype"
        assert "bfloat16" in check.GATED_KV


class TestTheTwoRankHarnessStaysHonest:
    """The 2-rank validation is the only place the fix is exercised with a ring
    before the campaign pays for 8 ranks, so its inputs have to be pinned."""

    def test_its_tolerance_is_the_gate_s_tolerance(self):
        """Two copies of one number drift, and a laxer copy here would certify a
        probe the gate then rejects."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import mlsys_cap_smoke_check as check

        runner = (REPO_ROOT / "scripts/mlsys_probe_2rank_check.sh").read_text()
        m = re.search(r"^TOL_BF16=([0-9.]+)", runner, re.M)
        assert m, "the runner does not state a TOL_BF16"
        assert float(m.group(1)) == check.TOL_BF16, (
            f"runner TOL_BF16={m.group(1)} but the checker uses "
            f"{check.TOL_BF16}")

    def test_it_requires_the_measured_kv_to_be_bf16(self):
        """A kv_quant override that silently did nothing would otherwise read as
        a passing run -- the exact trap the 1x noise-floor harness fell into."""
        runner = (REPO_ROOT / "scripts/mlsys_probe_2rank_check.sh").read_text()
        assert '!= "bfloat16"' in runner

    def test_it_requires_the_control_to_be_exactly_zero(self):
        """The control is the same computation twice. Anything but 0.0 means the
        run is not deterministic and the rest of the numbers mean nothing."""
        runner = (REPO_ROOT / "scripts/mlsys_probe_2rank_check.sh").read_text()
        assert '"control_max_abs_delta"' in runner
        assert "expected 0.0" in runner

    def test_it_asserts_the_rank_count_it_was_asked_for(self):
        """The whole point is the rank count, and it is now whatever GPU count
        the instance has -- so it must be compared against the nproc the run was
        launched with, not a hardcoded 2. A probe that silently ran on ONE rank
        looks identical to a success otherwise."""
        runner = (REPO_ROOT / "scripts/mlsys_probe_2rank_check.sh").read_text()
        assert '!= want_ranks' in runner, (
            "the rank-count assertion is not tied to the launched nproc")
        assert 'want_ranks = int(sys.argv[6])' in runner
        # and the default is still the cheapest configuration that can expose
        # the bug
        assert "NPROC=${3:-${MLSYS_PROBE2_NPROC:-2}}" in runner

    def test_the_config_probes_only_a_bf16_pair_at_short_context(self):
        import yaml

        cfg = yaml.safe_load(
            (REPO_ROOT / "configs/mlsys_probe_2rank.yml").read_text())
        levels = cfg["PROBE_2RANK"]["levels"]
        assert len(levels) == 2, "one pair, spec and target-only"
        for lv in levels:
            assert lv.get("teacher_forced_check") is True
            assert lv.get("kv_quant") is False, "the gated pair is the bf16 one"
            assert lv["context_length"] == 8192, (
                "rank participation does not depend on context length; paying "
                "for 128k here would pay for the campaign's question")

    def test_no_campaign_path_reads_the_validation_config(self):
        """It must not be reachable from the manifest or the watcher, or a
        validation config could be run as a campaign stage."""
        for name in ("scripts/mlsys_manifest.sh", "scripts/mlsys_watch_and_run.sh",
                     "configs/mlsys_manifest.yml"):
            src = (REPO_ROOT / name).read_text()
            assert "mlsys_probe_2rank" not in src, (
                f"{name} references the validation config")


class TestWorkersRunTheSameInterpreter:
    def test_torchrun_is_not_looked_up_on_path(self):
        """A worker must run THIS interpreter.

        `torchrun` from PATH is a different interpreter whenever PATH does not
        put this environment first, and the children then run against that
        interpreter's site-packages. On a validation pod on 2026-10-09 the
        symptom was workers dying with `ModuleNotFoundError: No module named
        'transformers'` while the parent -- which had just imported transformers
        to build the run -- reported only that the row errored. The console
        script and `python -m torch.distributed.run` are equivalent.
        """
        src = inspect.getsource(run_experiment.execute_run)
        spawn = src[src.index("if nproc > 1:"):src.index("else:", src.index("if nproc > 1:"))]
        assert '"torchrun"' not in spawn, (
            "the multi-rank spawn still resolves torchrun from PATH")
        assert '"torch.distributed.run"' in spawn
        assert "sys.executable" in spawn, (
            "the spawn must use the interpreter that is running this process")
