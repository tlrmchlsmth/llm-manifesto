"""Cluster YAML loading and concrete cluster facts used while rendering pods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .equations import render_mapping
from .images import DEFAULT_IMAGES, ImageCatalog


@dataclass(frozen=True)
class LlmdImages:
    release: str
    epp: str
    routing_sidecar: str

    @classmethod
    def from_config(cls, data: dict[str, Any], catalog: ImageCatalog = DEFAULT_IMAGES) -> "LlmdImages":
        release = str(data.get("release", catalog.get("llm_d.release")))
        templates = data.get("images", {})
        return cls(
            release=release,
            epp=templates.get("epp", catalog.get("llm_d.epp", release=release)).format(release=release),
            routing_sidecar=templates.get(
                "routing_sidecar",
                catalog.get("llm_d.routing_sidecar", release=release),
            ).format(release=release),
        )


@dataclass(frozen=True)
class Cluster:
    name: str
    gpus_per_node: int
    lustre_pvc: str
    local_nvme_path: str
    shm_size: str
    ucx_net_devices: str
    llm_d: LlmdImages
    fabric_default_profile: str = "standard"
    fabric_role_profiles: dict[str, str] | None = None
    fabric_default_env: dict[str, str] | None = None
    fabric_profiles: dict[str, dict[str, Any]] | None = None
    imex_resource_claim_template: str | None = None
    hf_cache_host_path: str | None = None
    jit_cache_host_path: str | None = None
    hf_home: str = "/mnt/local/hf_cache"
    rdma_resource_name: str | None = None
    rdma_resource_value: str = "1"
    user_root_template: str = "/mnt/lustre/{user}"
    log_root_template: str = "/mnt/lustre/{user}/logs"
    logging_pvc: str | None = None
    logging_mount_path: str = "/mnt/logs"
    cache_root_template: str = "/mnt/lustre/{user}/jit-cache/{gpu_arch}/{cuda}/{vllm_version}/{release}"
    dev_venv_template: str = "/mnt/lustre/{user}/vllm-venv"
    dev_source_template: str = "/mnt/lustre/{user}/vllm-dev"
    images: ImageCatalog = DEFAULT_IMAGES

    def base_volumes(self) -> list[dict]:
        volumes = [
            {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": self.shm_size}},
            {"name": "sys", "hostPath": {"path": "/sys", "type": "Directory"}},
            {"name": "proc", "hostPath": {"path": "/proc", "type": "Directory"}},
        ]
        if self.hf_cache_host_path and self.jit_cache_host_path:
            volumes.extend(
                [
                    {
                        "name": "hf-cache",
                        "hostPath": {"path": self.hf_cache_host_path, "type": "DirectoryOrCreate"},
                    },
                    {
                        "name": "jit-cache",
                        "hostPath": {"path": self.jit_cache_host_path, "type": "DirectoryOrCreate"},
                    },
                ]
            )
        else:
            volumes.extend(
                [
                    {"name": "lustre", "persistentVolumeClaim": {"claimName": self.lustre_pvc}},
                    {"name": "local-nvme", "hostPath": {"path": self.local_nvme_path, "type": "Directory"}},
                ]
            )
        if self.logging_pvc and self.logging_mount_path not in self._base_mount_paths():
            volumes.append({"name": "logs", "persistentVolumeClaim": {"claimName": self.logging_pvc}})
        return volumes

    def volume_mounts(self) -> list[dict]:
        mounts = self._base_volume_mounts()
        if self.logging_pvc and self.logging_mount_path not in self._base_mount_paths():
            mounts.append({"name": "logs", "mountPath": self.logging_mount_path})
        return mounts

    def _base_mount_paths(self) -> set[str]:
        return {mount["mountPath"] for mount in self._base_volume_mounts()}

    def _base_volume_mounts(self) -> list[dict]:
        if self.hf_cache_host_path and self.jit_cache_host_path:
            return [
                {"name": "dshm", "mountPath": "/dev/shm"},
                {"name": "hf-cache", "mountPath": "/var/cache/huggingface"},
                {"name": "jit-cache", "mountPath": "/var/cache/vllm"},
            ]
        return [
            {"name": "dshm", "mountPath": "/dev/shm"},
            {"name": "lustre", "mountPath": "/mnt/lustre"},
            {"name": "local-nvme", "mountPath": "/mnt/local"},
        ]

    def user_root(self, *, user: str, release: str) -> str:
        return self._format_path(self.user_root_template, user=user, release=release)

    def log_root(self, *, user: str, release: str) -> str:
        return self._format_path(self.log_root_template, user=user, release=release)

    def cache_root(self, *, user: str, release: str, gpu_arch: str, cuda: str, vllm_version: str) -> str:
        return self._format_path(
            self.cache_root_template,
            user=user,
            release=release,
            gpu_arch=gpu_arch,
            cuda=cuda,
            vllm_version=vllm_version,
        )

    def dev_venv(self, *, user: str, release: str) -> str:
        return self._format_path(self.dev_venv_template, user=user, release=release)

    def dev_source(self, *, user: str, release: str) -> str:
        return self._format_path(self.dev_source_template, user=user, release=release)

    def with_path_overrides(
        self,
        *,
        user_root: str | None = None,
        log_root: str | None = None,
        cache_root: str | None = None,
        dev_venv: str | None = None,
        dev_source: str | None = None,
    ) -> "Cluster":
        return Cluster(
            name=self.name,
            gpus_per_node=self.gpus_per_node,
            lustre_pvc=self.lustre_pvc,
            local_nvme_path=self.local_nvme_path,
            shm_size=self.shm_size,
            ucx_net_devices=self.ucx_net_devices,
            llm_d=self.llm_d,
            fabric_default_profile=self.fabric_default_profile,
            fabric_role_profiles=dict(self.fabric_role_profiles or {}),
            fabric_default_env=dict(self.fabric_default_env or {}),
            fabric_profiles=dict(self.fabric_profiles or {}),
            imex_resource_claim_template=self.imex_resource_claim_template,
            hf_cache_host_path=self.hf_cache_host_path,
            jit_cache_host_path=self.jit_cache_host_path,
            hf_home=self.hf_home,
            rdma_resource_name=self.rdma_resource_name,
            rdma_resource_value=self.rdma_resource_value,
            user_root_template=user_root or self.user_root_template,
            log_root_template=log_root or self.log_root_template,
            logging_pvc=self.logging_pvc,
            logging_mount_path=self.logging_mount_path,
            cache_root_template=cache_root or self.cache_root_template,
            dev_venv_template=dev_venv or self.dev_venv_template,
            dev_source_template=dev_source or self.dev_source_template,
            images=self.images,
        )

    def _format_path(self, template: str, **values: str) -> str:
        return template.format(**values)

    def fabric_profile_for(self, *, topology: str, role_name: str, expert_parallel: bool) -> str:
        if not expert_parallel:
            return self.fabric_default_profile
        return (self.fabric_role_profiles or {}).get(role_name, self.fabric_default_profile)

    def fabric_env(self, profile: str, context: dict | None = None) -> dict[str, str]:
        format_context = {"ucx_net_devices": self.ucx_net_devices}
        env = {key: str(value).format(**format_context) for key, value in (self.fabric_default_env or {}).items()}
        profile_config = (self.fabric_profiles or {}).get(profile, {})
        env |= {key: str(value) for key, value in profile_config.get("env", {}).items()}
        if profile_config.get("computed_env"):
            env |= {
                key: str(value)
                for key, value in render_mapping(profile_config["computed_env"], context or {}).items()
            }
        return env


def load_cluster(path: str | Path) -> Cluster:
    with Path(path).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    paths = data.get("paths", {})
    dev = data.get("dev", {})
    fabric = data.get("fabric", {})
    cache = data.get("cache", {})
    rdma = data.get("rdma", {})
    logging = data.get("logging", {})
    user_root = paths.get("user_root", "/mnt/lustre/{user}")
    return Cluster(
        name=data["name"],
        gpus_per_node=int(data.get("gpus_per_node", 4)),
        lustre_pvc=data["storage"]["lustre_pvc"],
        local_nvme_path=data["storage"].get("local_nvme_path", "/mnt/numa0"),
        shm_size=data.get("pod_defaults", {}).get("shm_size", "2Gi"),
        ucx_net_devices=fabric["ucx_net_devices"],
        llm_d=LlmdImages.from_config(data.get("llm_d", {}), DEFAULT_IMAGES),
        fabric_default_profile=fabric.get("default_profile", "standard"),
        fabric_role_profiles=fabric.get("expert_parallel_profiles", {}),
        fabric_default_env=fabric.get("default_env", {}),
        fabric_profiles=fabric.get("profiles", {}),
        imex_resource_claim_template=fabric.get("imex_resource_claim_template"),
        hf_cache_host_path=cache.get("hf_host_path"),
        jit_cache_host_path=cache.get("jit_host_path"),
        hf_home=cache.get("hf_home", "/mnt/local/hf_cache"),
        rdma_resource_name=rdma.get("resource_name"),
        rdma_resource_value=str(rdma.get("value", "1")),
        user_root_template=user_root,
        log_root_template=logging.get("root", paths.get("log_root", f"{user_root}/logs")),
        logging_pvc=logging.get("pvc"),
        logging_mount_path=logging.get("mount_path", "/mnt/logs"),
        cache_root_template=paths.get(
            "cache_root",
            "/mnt/lustre/{user}/jit-cache/{gpu_arch}/{cuda}/{vllm_version}/{release}",
        ),
        dev_venv_template=dev.get("venv", "/mnt/lustre/{user}/vllm-venv"),
        dev_source_template=dev.get("source", "/mnt/lustre/{user}/vllm-dev"),
        images=DEFAULT_IMAGES,
    )
