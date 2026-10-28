"""解析显存，用来把放不进单卡的方案排除出可行集。

这不是实测剖面。峰值显存若已测到，留在样本里供对照，不参与这里的判断。
CP 按修订稿里的实现：KV 整份留在每张卡上，不因 CP 下降。
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import BlockSpec, Load, ModelSpec, Plan, Split


def _shard(nbytes: int, degree: int) -> int:
    if degree <= 1:
        return nbytes
    return (nbytes + degree - 1) // degree


@dataclass(frozen=True)
class MemoryParts:
    param_bytes: int
    kv_bytes: int
    activation_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.param_bytes + self.kv_bytes + self.activation_bytes


def _param_bytes(block: BlockSpec, split: Split) -> int:
    table = block.table_bytes
    if block.kind == "embedding" and split.kind == "embp":
        table = _shard(table, split.degree)
    dense = block.param_bytes
    if split.kind == "tp" or split.kind == "pp":
        dense = _shard(dense, split.degree)
    elif split.kind == "ep" and block.kind == "expert":
        dense = _shard(dense, split.degree)
    return table + dense


def _kv_bytes(block: BlockSpec, load: Load, split: Split, elem_bytes: int) -> int:
    if block.kind != "attention":
        return 0
    length = load.lengths[block.name]
    total = 2 * block.n_layers * load.batch * length * block.d_model * elem_bytes
    if split.kind in ("tp", "pp"):
        return _shard(total, split.degree)
    return total


def _activation_bytes(block: BlockSpec, load: Load, split: Split, elem_bytes: int) -> int:
    if block.kind in ("embedding", "cross"):
        count = load.n_ids[block.name]
    else:
        count = load.lengths[block.name]
    total = block.n_layers * load.batch * count * block.d_model * elem_bytes
    if split.kind in ("tp", "cp", "pp", "ep"):
        return _shard(total, split.degree)
    return total


def memory_parts(spec: ModelSpec, load: Load, plan: Plan) -> MemoryParts:
    param = kv = activation = 0
    for block in spec.blocks:
        split = plan.split_of(block.name)
        param += _param_bytes(block, split)
        kv += _kv_bytes(block, load, split, spec.elem_bytes)
        activation += _activation_bytes(block, load, split, spec.elem_bytes)
    return MemoryParts(param, kv, activation)
