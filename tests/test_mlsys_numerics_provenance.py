"""The precision a row reports must be the precision the run USED.

Every comparison in the campaign now rests on one claim: RASD runs FP4 weights
with an NF4 KV cache, and vLLM cannot reproduce that cache -- which is why the
vLLM cross-check reports token agreement descriptively instead of scoring it. A
`weight_precision`/`kv_dtype` column derived from the run's CONFIG reports the
instruction rather than the fact, and this project has already shipped a path
where the instruction was not carried out (`quantize_target` on CPU/MPS logs a
warning and loads dense weights). The whole vLLM reasoning would inherit that
error silently.

So the columns are read off the loaded model (`is_loaded_in_4bit` and its
`bnb_4bit_quant_type`) and the live cache object, and the spellings are
normalized so two engines' precision can be compared at all.
"""
from __future__ import annotations

import pytest
import torch

from src.models.nf4_dynamic_cache import NF4DynamicCache
from src.models.rasd_inference import (
    RASDConfig, detect_kv_precision, detect_weight_precision,
    normalize_dtype_name, numerics_report,
)


class _QC:
    def __init__(self, qtype):
        self.bnb_4bit_quant_type = qtype


class _Cfg:
    def __init__(self, qtype=None):
        if qtype is not None:
            self.quantization_config = _QC(qtype)


class _Param:
    def __init__(self, dtype):
        self.dtype = dtype


class _Dense:
    is_loaded_in_4bit = False

    def __init__(self, dtype=torch.bfloat16):
        self._p = _Param(dtype)

    def parameters(self):
        return iter([self._p])


class _FourBit:
    """What transformers leaves on the model after a bnb 4-bit load."""

    is_loaded_in_4bit = True

    def __init__(self, qtype="fp4"):
        self.config = _Cfg(qtype)


class _EightBit:
    is_loaded_in_4bit = False
    is_loaded_in_8bit = True


def test_a_4bit_load_reports_its_own_storage_type():
    assert detect_weight_precision(_FourBit("fp4")) == "fp4"
    assert detect_weight_precision(_FourBit("nf4")) == "nf4"


def test_a_4bit_load_that_declares_no_type_is_unknown_never_guessed():
    """An unstated 4-bit type is `unknown`, not a guess.

    transformers' `BitsAndBytesConfig` defaults to fp4, so a fallback of "nf4"
    would be wrong in the common case -- and wrong toward a value the cap smoke
    ACCEPTS, which is how a mislabelled row reaches a table. `unknown` fails
    that assertion, which is the honest outcome for a precision nobody recorded.
    """
    assert detect_weight_precision(_FourBit(None)) == "unknown"
    assert detect_weight_precision(
        type("M", (), {"is_loaded_in_4bit": True, "config": _Cfg()})()) == "unknown"
    # ... and it must not be confused with the fp4/nf4 the smoke accepts.
    for accepted in ("fp4", "nf4"):
        assert detect_weight_precision(_FourBit(None)) != accepted


def test_a_dense_load_reports_its_parameter_dtype():
    assert detect_weight_precision(_Dense(torch.bfloat16)) == "bfloat16"
    assert detect_weight_precision(_Dense(torch.float16)) == "float16"
    assert detect_weight_precision(_Dense(torch.float32)) == "float32"


def test_the_4bit_flag_is_checked_before_the_parameter_dtype():
    """A 4-bit model's parameters are uint8 storage behind a dequantize hook.

    Reading the parameter dtype first would report `uint8` (or the compute
    dtype) and lose the fp4/nf4 distinction -- the exact axis the vLLM
    comparison turns on.
    """
    model = _FourBit("fp4")
    model.parameters = lambda: iter([_Param(torch.uint8)])
    assert detect_weight_precision(model) == "fp4"


def test_an_8bit_load_is_distinguishable_from_a_4bit_one():
    assert detect_weight_precision(_EightBit()) == "int8"


def test_there_is_no_model_and_no_cache_case_that_must_not_guess():
    assert detect_weight_precision(None) == ""
    assert detect_kv_precision(None) == ""


def test_the_kv_precision_comes_from_the_cache_class():
    cache = NF4DynamicCache(block_size=64, dtype=torch.bfloat16,
                            bf16_prefix_size=0, update_chunk_size=0)
    assert detect_kv_precision(cache) == "nf4", (
        "kv_quant=True with a cache the model did not use would otherwise be "
        "reported as quantized")


def test_a_legacy_bf16_tuple_is_reported_as_bf16():
    k = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    v = torch.zeros(1, 2, 4, 8, dtype=torch.bfloat16)
    assert detect_kv_precision(((k, v),)) == "bfloat16"


def test_an_unnamable_cache_says_unknown_rather_than_guessing():
    class _Odd:
        def get_seq_length(self, layer_idx=0):
            return 7

    assert detect_kv_precision(_Odd()) == "unknown"


def test_dtype_spellings_are_normalized_so_engines_can_be_compared():
    """`bf16` and `bfloat16` are one dtype, and a table must show that."""
    assert normalize_dtype_name(torch.bfloat16) == \
        normalize_dtype_name("bf16") == normalize_dtype_name("bfloat16")
    assert normalize_dtype_name("fp16") == normalize_dtype_name(torch.float16)
    assert normalize_dtype_name(None) == ""


@pytest.mark.parametrize("dtype,alias", [
    (torch.bfloat16, "bf16"), (torch.float16, "fp16"), (torch.float32, "fp32"),
])
def test_every_alias_agrees_with_the_canonical_name(dtype, alias):
    assert normalize_dtype_name(dtype) == normalize_dtype_name(alias)


def test_the_campaign_configuration_is_fp4_weights_and_an_nf4_cache():
    """The pair the plan's vLLM reasoning depends on, as one assertion."""
    r = numerics_report(_FourBit("fp4"),
                        NF4DynamicCache(block_size=64, dtype=torch.bfloat16,
                                        bf16_prefix_size=0,
                                        update_chunk_size=0))
    assert r == {"weight_precision": "fp4", "kv_dtype": "nf4"}


def test_config_flags_are_not_what_is_reported():
    """A config claiming 4-bit, with dense weights loaded, must not report fp4.

    This is the failure the column exists to catch: the loader warns and skips
    quantization off CUDA, so a config-derived label would report `fp4` for a
    run that actually ran bf16.
    """
    cfg = RASDConfig(quantize_target=True, kv_quant=True)
    assert cfg.quantize_target is True and cfg.kv_quant is True
    r = numerics_report(_Dense(torch.bfloat16), ((torch.zeros(1, 1, 2, 4,
                                                             dtype=torch.bfloat16),) * 2,))
    assert r["weight_precision"] != "fp4", (
        "the reported precision followed the config, not the loaded model")
    assert r["kv_dtype"] != "nf4", (
        "the reported KV dtype followed the config, not the live cache")


def test_a_dict_shaped_quantization_config_is_read_too():
    """A config round-tripped through `config.json` arrives as a mapping."""
    class _M:
        is_loaded_in_4bit = True

        class _C:
            quantization_config = {"bnb_4bit_quant_type": "fp4"}
        config = _C()

    assert detect_weight_precision(_M()) == "fp4"


def test_an_attribute_beats_a_mapping_lookup_on_a_dict_subclass():
    """The precedence that matters: reading the mapping first reports nf4."""
    class _QC(dict):
        bnb_4bit_quant_type = "fp4"

    class _M:
        is_loaded_in_4bit = True

        class _C:
            quantization_config = _QC()
        config = _C()

    assert detect_weight_precision(_M()) == "fp4", (
        "a dict that carries the value as an attribute was read as 'unset', so "
        "an fp4 model would be labelled nf4")
