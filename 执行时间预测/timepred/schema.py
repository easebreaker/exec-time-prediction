"""样本、结构和切分方案。

一条已实现的前向是一个样本：负载 x、方案 p、端到端时间。
有分段计时则带上各块时间。峰值显存只随样本保存，可行集另用解析显存。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

BLOCK_KINDS = ("embedding", "cross", "attention", "token_mixer", "ffn", "expert")
SPLIT_KINDS = ("none", "dp", "tp", "pp", "ep", "embp", "cp")
# 与修订稿里的对应关系：切分只改它切得到的那类张量。
SPLIT_BLOCK_KINDS = {
    "none": BLOCK_KINDS,
    "dp": BLOCK_KINDS,
    "tp": ("attention", "ffn", "cross", "token_mixer"),
    "pp": ("attention", "ffn", "expert", "cross", "token_mixer"),
    "ep": ("expert",),
    "embp": ("embedding",),
    "cp": ("attention",),
}


class FrameworkError(ValueError):
    """输入、样本或参数不满足这个预测框架的约定。"""


@dataclass(frozen=True)
class BlockSpec:
    """结构 S 里的一个计算块。"""

    name: str
    kind: str
    d_model: int
    n_layers: int = 1
    param_bytes: int = 0
    table_bytes: int = 0

    def __post_init__(self) -> None:
        if self.kind not in BLOCK_KINDS:
            raise FrameworkError(f"未知的计算块类型 {self.kind}")
        if self.d_model < 1 or self.n_layers < 1:
            raise FrameworkError(f"{self.name} 的宽度和层数至少为 1")
        if self.param_bytes < 0 or self.table_bytes < 0:
            raise FrameworkError(f"{self.name} 的字节数不能为负")


@dataclass(frozen=True)
class ModelSpec:
    """已经测过的那一个模型。资源在这批样本里视为常数。"""

    blocks: tuple[BlockSpec, ...]
    capacity_bytes: int
    elem_bytes: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        names = [block.name for block in self.blocks]
        if len(names) != len(set(names)):
            raise FrameworkError("计算块名称重复")
        if not names:
            raise FrameworkError("至少要有一个计算块")
        if self.capacity_bytes <= 0 or self.elem_bytes <= 0:
            raise FrameworkError("单卡容量和元素字节数要为正")

    def block(self, name: str) -> BlockSpec:
        for block in self.blocks:
            if block.name == name:
                return block
        raise FrameworkError(f"结构里没有计算块 {name}")


@dataclass(frozen=True)
class Split:
    """一个块上的切分。degree 是并行度。流水并行另给 micro-batch 数。"""

    kind: str
    degree: int = 1
    n_microbatches: int = 1

    def __post_init__(self) -> None:
        if self.kind not in SPLIT_KINDS:
            raise FrameworkError(f"未知的切分 {self.kind}")
        if self.degree < 1 or self.n_microbatches < 1:
            raise FrameworkError("并行度和 micro-batch 数至少为 1")
        if self.kind == "none" and (self.degree != 1 or self.n_microbatches != 1):
            raise FrameworkError("不切分的并行度为 1")
        if self.kind != "pp" and self.n_microbatches != 1:
            raise FrameworkError("只有流水并行记录 micro-batch 数")
        if self.kind in ("tp", "pp", "ep", "embp", "cp") and self.degree < 2:
            raise FrameworkError(f"{self.kind} 的并行度至少为 2")


@dataclass(frozen=True)
class Plan:
    """方案 p。没有写到的块视为不切分。"""

    splits: Mapping[str, Split]

    def __post_init__(self) -> None:
        object.__setattr__(self, "splits", dict(self.splits))

    def split_of(self, block_name: str) -> Split:
        return self.splits.get(block_name, Split("none"))


@dataclass(frozen=True)
class Load:
    """负载 x。各块可以有自己的序列长度。"""

    batch: int
    lengths: Mapping[str, int]
    n_ids: Mapping[str, int] | None = None
    expert_frac: float = 1.0
    n_groups: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.batch < 1:
            raise FrameworkError("batch 至少为 1")
        if not 0.0 <= self.expert_frac <= 1.0:
            raise FrameworkError("expert 实际执行份额要在 0 到 1 之间")
        lengths = dict(self.lengths)
        n_ids = dict(self.n_ids or {})
        n_groups = dict(self.n_groups or {})
        for name, length in lengths.items():
            if length < 1:
                raise FrameworkError(f"{name} 的序列长度至少为 1")
        for name, count in n_ids.items():
            if count < 1:
                raise FrameworkError(f"{name} 的 id 数至少为 1")
        for name, groups in n_groups.items():
            if groups < 1:
                raise FrameworkError(f"{name} 的组数至少为 1")
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "n_ids", n_ids)
        object.__setattr__(self, "n_groups", n_groups)

    def key(self) -> tuple:
        """同一长度、同一 batch 的各次切分属于同一负载组合。"""
        lengths = tuple(sorted(self.lengths.items()))
        n_ids = tuple(sorted(self.n_ids.items()))
        n_groups = tuple(sorted(self.n_groups.items()))
        return (self.batch, self.expert_frac, lengths, n_ids, n_groups)


@dataclass(frozen=True)
class Sample:
    """一次已经实现并计时的前向。"""

    load: Load
    plan: Plan
    time_s: float
    block_times_s: Mapping[str, float] | None = None
    peak_mem_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.time_s <= 0.0:
            raise FrameworkError("端到端时间要为正")
        if self.peak_mem_bytes is not None and self.peak_mem_bytes < 0:
            raise FrameworkError("峰值显存不能为负")
        if self.block_times_s is not None:
            times = dict(self.block_times_s)
            for name, value in times.items():
                if value < 0.0:
                    raise FrameworkError(f"{name} 的分段时间不能为负")
            object.__setattr__(self, "block_times_s", times)


def plan_token(plan: Plan) -> str:
    parts = []
    for name in sorted(plan.splits):
        split = plan.splits[name]
        if split.kind == "none":
            continue
        piece = f"{name}:{split.kind}:{split.degree}"
        if split.kind == "pp":
            piece += f":{split.n_microbatches}"
        parts.append(piece)
    return ";".join(parts) if parts else "none"


def active_kinds(spec: ModelSpec, plan: Plan) -> frozenset[str]:
    kinds = set()
    for block in spec.blocks:
        kind = plan.split_of(block.name).kind
        if kind not in ("none", "dp"):
            kinds.add(kind)
    return frozenset(kinds)


def is_control(spec: ModelSpec, plan: Plan) -> bool:
    """不切分或只做数据并行。单次前向的预测时间取对照。"""
    return not active_kinds(spec, plan)


_KIND_LABEL = {
    "none": "不切分",
    "dp": "DP",
    "tp": "TP",
    "pp": "PP",
    "ep": "EP",
    "embp": "EmbP",
    "cp": "CP",
    "embedding": "查表",
    "cross": "交叉",
    "attention": "注意力",
    "token_mixer": "token mixer",
    "ffn": "FFN",
    "expert": "Expert",
}


def kind_label(kind: str) -> str:
    return _KIND_LABEL.get(kind, kind)


def ensure_plan(spec: ModelSpec, plan: Plan) -> None:
    """方案里的块必须属于这个模型，并且切分用在对得上的张量上。"""
    for name, split in plan.splits.items():
        block = spec.block(name)
        if block.kind not in SPLIT_BLOCK_KINDS[split.kind]:
            raise FrameworkError(
                f"{name} 是{kind_label(block.kind)}，不能用 {kind_label(split.kind)}"
            )
