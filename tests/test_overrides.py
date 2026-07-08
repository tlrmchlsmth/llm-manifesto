"""Tests for YAML spec inheritance and role override composition."""

from pathlib import Path

import pytest

from manifesto.cluster import load_cluster
from manifesto.overrides import merge_overrides
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "oci-gb200.yaml")


def test_role_map_overrides_merge_by_name():
    merged = merge_overrides(
        {
            "roles": [
                {"name": "decode", "lws": {"size": 2, "replicas": 1}, "vllm": {"moe_backend": "mega"}},
                {"name": "prefill", "lws": {"size": 2, "replicas": 2}},
            ]
        },
        {"roles": {"decode": {"lws": {"size": 4}, "vllm": {"all2all_backend": "flashinfer"}}}},
    )

    assert merged["roles"][0]["lws"] == {"size": 4, "replicas": 1}
    assert merged["roles"][0]["vllm"] == {
        "moe_backend": "mega",
        "all2all_backend": "flashinfer",
    }
    assert merged["roles"][1]["lws"] == {"size": 2, "replicas": 2}


def test_role_map_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        merge_overrides({"roles": [{"name": "decode"}]}, {"roles": {"prefill": {"lws": {"replicas": 2}}}})


@pytest.mark.parametrize(
    ("config", "prefill_replicas", "decode_size", "decode_replicas", "decode_dp", "moe_backend", "all2all"),
    [
        ("1P-EP8-1D-EP8.yaml", 1, 2, 1, 8, "deep_gemm_mega_moe", None),
        ("2P-EP8-1D-EP8.yaml", 2, 2, 1, 8, "deep_gemm_mega_moe", None),
        ("3P-EP8-1D-EP8.yaml", 3, 2, 1, 8, "deep_gemm_mega_moe", None),
        ("3P-EP8-2D-EP16.yaml", 3, 4, 2, 16, "auto", "flashinfer_nvlink_one_sided"),
    ],
)
def test_deepseek_v4_llmd_guide_variants_expand(
    config: str,
    prefill_replicas: int,
    decode_size: int,
    decode_replicas: int,
    decode_dp: int,
    moe_backend: str,
    all2all: str | None,
):
    spec = load_spec(ROOT / "models" / "deepseek-v4" / config, CLUSTER)
    decode = spec.role("decode")
    prefill = spec.role("prefill")

    assert spec.topology == "pd"
    assert spec.model.id == "deepseek-ai/DeepSeek-V4-Pro"
    assert spec.model.image == "vllm/vllm-openai:v0.23.0"

    assert prefill.lws.size == 2
    assert prefill.lws.replicas == prefill_replicas
    assert prefill.tensor_parallel_size == 1
    assert prefill.data_parallel.enabled is True
    assert prefill.data_parallel.local_size == 4
    assert prefill.kv_transfer_config["kv_role"] == "kv_producer"

    assert decode.lws.size == decode_size
    assert decode.lws.replicas == decode_replicas
    assert decode.tensor_parallel_size == 1
    assert decode.data_parallel.enabled is True
    assert decode.data_parallel.local_size == decode_dp // decode_size
    assert decode.kv_transfer_config["kv_role"] == "kv_consumer"
    assert decode.vllm_args["moe_backend"] == moe_backend
    assert decode.vllm_args.get("all2all_backend") == all2all
