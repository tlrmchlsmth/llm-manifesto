"""Derive local and global TP/DP layout from a role's parallelism settings."""

from __future__ import annotations

from dataclasses import dataclass

from .spec import RoleSpec


@dataclass(frozen=True)
class ParallelLayout:
    tp_world_size: int
    tp_local_size: int
    dp_enabled: bool
    dp_requested: int
    dp_local_size: int
    dp_world_size: int


def parallel_layout(role: RoleSpec) -> ParallelLayout:
    parallelism = role.parallelism
    tp_local_size = _local_tp_size(role)

    if parallelism.dp_enabled:
        dp_local_size = max(1, parallelism.dp_size // role.lws.size)
        dp_world_size = role.lws.size * dp_local_size
    else:
        dp_local_size = max(1, role.gpus_per_pod // tp_local_size)
        dp_world_size = 1

    return ParallelLayout(
        tp_world_size=parallelism.tp,
        tp_local_size=tp_local_size,
        dp_enabled=parallelism.dp_enabled,
        dp_requested=parallelism.dp_size,
        dp_local_size=dp_local_size,
        dp_world_size=dp_world_size,
    )


def _local_tp_size(role: RoleSpec) -> int:
    if role.parallelism.tp <= role.gpus_per_pod:
        return role.parallelism.tp
    return max(1, role.parallelism.tp // role.lws.size)
