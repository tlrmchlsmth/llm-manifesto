"""Pydantic models and loader for user-authored deployment YAML specs."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .cluster import Cluster
from .images import apply_image_refs
from .overrides import load_spec_data
from .parallelism import parallel_layout


class TopologyKind(StrEnum):
    AGGREGATED = "aggregated"
    PD = "pd"


class RoutingKind(StrEnum):
    LOAD_AWARE = "load_aware"
    PD = "pd"
    DISABLED = "disabled"


class DpLoadBalancing(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"


# GPU count assumed when a spec is loaded without a cluster profile to infer from.
DEFAULT_GPUS_PER_POD = 4


class LwsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size: int = Field(1, ge=1)
    replicas: int = Field(1, ge=1)


class ParallelismSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tp: int = Field(1, ge=1)
    dp: int | bool | None = None
    ep: bool = False
    gpus: int | None = Field(None, ge=1, validation_alias=AliasChoices("gpus", "gpus_per_node"))

    @field_validator("dp")
    @classmethod
    def validate_dp(cls, value: int | bool | None) -> int | bool | None:
        if value is True:
            raise ValueError("dp: true is ambiguous; use a global size or false")
        if isinstance(value, int) and not isinstance(value, bool) and value < 1:
            raise ValueError("dp must be >= 1")
        return value

    @property
    def dp_size(self) -> int:
        """Requested global data-parallel size; 1 means data parallelism is off."""
        if self.dp is None or self.dp is False:
            return 1
        return self.dp

    @property
    def dp_enabled(self) -> bool:
        return self.dp_size > 1


class ResourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu: str = "32"
    memory: str = "512Gi"
    gpus: int = Field(4, ge=0)
    ephemeral_storage: str = "128Gi"


class RoleSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    workload_name: str | None = None
    lws: LwsSpec = Field(default_factory=LwsSpec)
    parallelism: ParallelismSpec = Field(default_factory=ParallelismSpec)
    serving_port_base: int = 8000
    backend_port_base: int | None = None
    routing_proxy: bool = False
    dp_load_balancing: DpLoadBalancing = DpLoadBalancing.INTERNAL
    kv_transfer_config: dict[str, Any] | None = None
    vllm_args: dict[str, Any] = Field(
        default_factory=dict, validation_alias=AliasChoices("vllm", "vllm_args")
    )
    env: dict[str, str] = Field(default_factory=dict)
    pre_launch: list[str] = Field(default_factory=list)
    vars: dict[str, Any] = Field(default_factory=dict)
    computed: dict[str, dict[str, Any]] = Field(default_factory=dict)
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    shm_size: str | None = None

    @property
    def gpus_per_pod(self) -> int:
        if self.parallelism.gpus is not None:
            return self.parallelism.gpus
        return DEFAULT_GPUS_PER_POD

    @model_validator(mode="after")
    def default_backend_port(self) -> "RoleSpec":
        if self.routing_proxy and self.backend_port_base is None:
            self.backend_port_base = 8200
        return self


class ModelSpec(BaseModel):
    id: str
    label: str | None = None
    image: str
    served_name: str | None = None
    hf_home: str | None = None

    @property
    def label_value(self) -> str:
        return self.label or self.id.rsplit("/", 1)[-1]


class RuntimeSpec(BaseModel):
    dev: bool = False
    dev_venv: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    pre_launch: list[str] = Field(default_factory=list)
    sidecars: list[str] = Field(default_factory=lambda: ["dcgm-exporter", "node-exporter"])


class RoutingSpec(BaseModel):
    kind: RoutingKind | None = None
    epp_image: str | None = None
    plugin_config: dict[str, Any] | None = None
    replicas: int = Field(1, ge=1)
    target_role: str | None = None


class CacheSpec(BaseModel):
    gpu_arch: str = "gb200"
    cuda: str = "cu13"
    key: str | None = None


class DeploymentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: str
    namespace: str = "default"
    topology: TopologyKind
    model: ModelSpec
    roles: list[RoleSpec]
    routing: RoutingSpec = Field(default_factory=RoutingSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    vars: dict[str, Any] = Field(default_factory=dict)

    @property
    def cache_key(self) -> str:
        if self.cache.key:
            return _safe_cache_key(self.cache.key)
        image = self.model.image
        if "@" in image:
            identity = image.rsplit("@", 1)[1]
        else:
            image_name = image.rsplit("/", 1)[-1]
            identity = image_name.rsplit(":", 1)[1] if ":" in image_name else "latest"
        return _safe_cache_key(identity)

    @field_validator("roles")
    @classmethod
    def require_unique_roles(cls, roles: list[RoleSpec]) -> list[RoleSpec]:
        names = [role.name for role in roles]
        if len(names) != len(set(names)):
            raise ValueError("role names must be unique")
        return roles

    @model_validator(mode="after")
    def apply_topology_defaults(self) -> "DeploymentSpec":
        if self.routing.kind is None:
            self.routing.kind = RoutingKind.PD if self.topology == TopologyKind.PD else RoutingKind.LOAD_AWARE
        if self.routing.target_role is None:
            self.routing.target_role = "decode"
        if self.topology == TopologyKind.PD:
            decode = self.role("decode")
            decode.dp_load_balancing = DpLoadBalancing.EXTERNAL
            decode.routing_proxy = True
            decode.serving_port_base = 8000
            decode.backend_port_base = 8200
        for role in self.roles:
            if role.routing_proxy and role.dp_load_balancing != DpLoadBalancing.EXTERNAL:
                raise ValueError(f"{role.name}: routing_proxy requires dp_load_balancing: external")
            if role.dp_load_balancing == DpLoadBalancing.EXTERNAL and _api_server_count(role.vllm_args) > 1:
                raise ValueError(
                    f"{role.name}: api_server_count > 1 is incompatible with external DP load balancing"
                )
        return self

    def role(self, name: str) -> RoleSpec:
        for role in self.roles:
            if role.name == name:
                return role
        raise KeyError(f"unknown role: {name}")

    def apply_cluster_defaults(self, cluster: Cluster) -> None:
        if self.model.hf_home is None:
            self.model.hf_home = cluster.hf_home
        for role in self.roles:
            if role.parallelism.gpus is None:
                role.parallelism.gpus = _infer_gpus_per_pod(role, cluster.gpus_per_node)
            if "gpus" not in role.resources.model_fields_set:
                role.resources.gpus = role.gpus_per_pod
            parallel_layout(role)


def _api_server_count(vllm_args: dict[str, Any]) -> int:
    value = vllm_args.get("api_server_count", vllm_args.get("api-server-count", 1))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _infer_gpus_per_pod(role: RoleSpec, cluster_gpus_per_node: int) -> int:
    parallelism = role.parallelism
    if parallelism.tp > cluster_gpus_per_node:
        tp_local = max(1, parallelism.tp // role.lws.size)
    else:
        tp_local = parallelism.tp

    if parallelism.dp_enabled:
        return tp_local * max(1, parallelism.dp_size // role.lws.size)
    return tp_local


def _safe_cache_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "latest"


def load_spec(path: str | Path, cluster: Cluster | None = None) -> DeploymentSpec:
    data = load_spec_data(path)
    data = apply_image_refs(data)
    spec = DeploymentSpec.model_validate(data)
    if cluster is not None:
        spec.apply_cluster_defaults(cluster)
    return spec
