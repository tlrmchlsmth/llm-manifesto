"""Structural tests for rendered Kubernetes objects and YAML serialization."""

from pathlib import Path

import yaml

from manifesto.cluster import load_cluster
from manifesto.render import render, render_to_yaml
from manifesto.spec import load_spec


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = load_cluster(ROOT / "clusters" / "oci-gb200.yaml")
CKS_H200 = load_cluster(ROOT / "clusters" / "cks-h200.yaml")
DEEPSEEK = "deepseek-v4/1P-EP8-1D-EP8.yaml"


def _objects(config: str) -> list[dict]:
    spec = load_spec(ROOT / "models" / config, CLUSTER)
    return render(spec, user="tester", cluster=CLUSTER)


def _find(objects: list[dict], kind: str, name_suffix: str | None = None) -> dict:
    for obj in objects:
        if obj["kind"] != kind:
            continue
        if name_suffix is None or obj["metadata"]["name"].endswith(name_suffix):
            return obj
    raise AssertionError(f"missing {kind} {name_suffix or ''}")


def test_rendered_yaml_parses():
    objects = _objects(DEEPSEEK)
    parsed = list(yaml.safe_load_all(render_to_yaml(objects)))

    assert len(parsed) == len(objects)


def test_dp_ports_feed_container_readiness_and_inferencepool():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    infpool = _find(objects, "InferencePool")

    assert [p["containerPort"] for p in container["ports"]] == [8100, 8200, 8201, 8202, 8203]
    readiness = container["readinessProbe"]["exec"]["command"][-1]
    assert "localhost:8000" in readiness
    assert "localhost:8003" in readiness
    assert container["startupProbe"]["httpGet"]["port"] == "dp-supervisor"
    assert infpool["apiVersion"] == "inference.networking.k8s.io/v1"
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}, {"number": 8001}, {"number": 8002}, {"number": 8003}]
    assert infpool["spec"]["endpointPickerRef"]["name"] == "tester-wide-ep-1p-ep8-1d-ep8-infpool-epp"
    script = container["args"][0]
    assert "DP_SIZE=8" in script
    assert "DP_SIZE=$((LWS_GROUP_SIZE * DP_SIZE_LOCAL))" not in script
    assert "--data-parallel-multi-port-external-lb" in script
    assert "--data-parallel-supervisor-port 8100" in script
    assert "--data-parallel-start-rank $START_RANK" in script
    assert "--data-parallel-rank" not in script


def test_no_dp_qwen_uses_single_port_and_no_dp_flags():
    objects = _objects("qwen/aggregated.yaml")
    lws = _find(objects, "LeaderWorkerSet", "decode")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    script = container["args"][0]
    infpool = _find(objects, "InferencePool")

    assert [p["containerPort"] for p in container["ports"]] == [8000]
    assert "--data-parallel-size" not in script
    assert "startupProbe" not in container
    assert infpool["spec"]["targetPorts"] == [{"number": 8000}]


def test_inferencepool_selector_is_instance_scoped():
    objects = _objects(DEEPSEEK)
    infpool = _find(objects, "InferencePool")

    selector = infpool["spec"]["selector"]["matchLabels"]
    assert selector["app.kubernetes.io/instance"] == "tester-wide-ep-1p-ep8-1d-ep8"
    assert selector["llm-d.ai/role"] == "decode"


def test_inferencepool_references_epp_service():
    objects = _objects(DEEPSEEK)
    service = _find(objects, "Service", "infpool-epp")

    assert service["spec"]["selector"]["app.kubernetes.io/component"] == "epp"
    assert service["spec"]["ports"][0]["targetPort"] == 9002


def test_epp_uses_current_config_file_flag():
    objects = _objects(DEEPSEEK)
    deployment = _find(objects, "Deployment", "infpool-epp")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]

    assert container["image"] == "ghcr.io/llm-d/llm-d-inference-scheduler:v0.8.0"
    assert "--config-file=/etc/epp/plugins.yaml" in args
    assert "--pool-name=tester-wide-ep-1p-ep8-1d-ep8-infpool" in args
    assert "--pool-namespace=default" in args
    assert not any(arg.startswith("--plugins-config-file") for arg in args)


def test_epp_uses_dedicated_service_account_and_rbac():
    objects = _objects(DEEPSEEK)
    service_account = _find(objects, "ServiceAccount", "infpool-epp")
    role = _find(objects, "Role", "infpool-epp-rbac")
    binding = _find(objects, "RoleBinding", "infpool-epp-rbac")
    deployment = _find(objects, "Deployment", "infpool-epp")

    assert deployment["spec"]["template"]["spec"]["serviceAccountName"] == service_account["metadata"]["name"]
    assert binding["subjects"] == [
        {"kind": "ServiceAccount", "name": service_account["metadata"]["name"], "namespace": "default"}
    ]
    assert binding["roleRef"]["name"] == role["metadata"]["name"]

    rules = {(tuple(rule["apiGroups"]), tuple(rule["resources"])): rule["verbs"] for rule in role["rules"]}
    assert rules[(("",), ("pods",))] == ["get", "list", "watch"]
    assert rules[(("inference.networking.k8s.io",), ("inferencepools",))] == ["get", "list", "watch"]
    assert rules[
        (
            ("inference.networking.x-k8s.io",),
            ("inferencemodelrewrites", "inferencemodels", "inferenceobjectives", "inferencepoolimports"),
        )
    ] == ["get", "list", "watch"]


def test_cks_h200_cluster_uses_coreweave_cache_and_rdma_settings():
    spec = load_spec(ROOT / "models" / "qwen" / "h200-aggregated.yaml", CKS_H200)
    assert spec.model.hf_home == "/var/cache/huggingface"
    objects = render(spec, user="tester", cluster=CKS_H200)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    pod_spec = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
    container = pod_spec["containers"][0]
    script = container["args"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}

    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}
    mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}

    assert volumes["hf-cache"]["hostPath"]["path"] == "/mnt/local/hf-cache"
    assert volumes["jit-cache"]["hostPath"]["path"] == "/mnt/local/jit-cache"
    assert mounts["hf-cache"] == "/var/cache/huggingface"
    assert mounts["jit-cache"] == "/var/cache/vllm"
    assert env["NCCL_IB_HCA"] == "ibp"
    assert env["NVSHMEM_HCA_PREFIX"] == "ibp"
    assert env["HF_HUB_CACHE"] == "/var/cache/huggingface"
    assert env["FLASHINFER_WORKSPACE_BASE"] == "/var/cache/vllm/flashinfer"
    assert "MAX_TOKENS" not in env
    assert "--max-num-batched-tokens" not in script
    assert "--max-num-seqs" not in script
    assert "--max-cudagraph-capture-size" not in script
    assert container["resources"]["requests"]["rdma/ib"] == "1"
    assert container["resources"]["limits"]["rdma/ib"] == "1"


def test_lws_uses_cluster_routing_sidecar_image():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "decode")
    init_container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["initContainers"][0]

    assert init_container["image"] == "ghcr.io/llm-d/llm-d-routing-sidecar:v0.8.0"


def test_routing_plugin_config_can_be_inline_override():
    spec = load_spec(ROOT / "models" / "qwen" / "aggregated.yaml", CLUSTER)
    spec.routing.plugin_config = {
        "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [{"type": "weighted-random-picker", "name": "custom-picker"}],
    }
    objects = render(spec, user="tester", cluster=CLUSTER)
    config = _find(objects, "ConfigMap", "epp-config")

    assert "custom-picker" in config["data"]["plugins.yaml"]
    assert "active-request-scorer" not in config["data"]["plugins.yaml"]


def test_prefill_launch_uses_global_tp_and_local_gpu_span():
    objects = _objects(DEEPSEEK)
    lws = _find(objects, "LeaderWorkerSet", "prefill")
    container = lws["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]["containers"][0]
    script = container["args"][0]

    assert "--tensor-parallel-size 1" in script
    assert "GPU_START=$((R * 1))" in script
