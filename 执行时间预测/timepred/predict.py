"""由基准时间、κ、τ、v 得到一次前向的预测时间。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .baseline import BlockBaseline, baseline_time
from .formula import compute_speedup, pipeline_factor, speedup_upper_bound
from .memory import memory_parts
from .schema import (
    FrameworkError,
    Load,
    ModelSpec,
    Plan,
    ensure_plan,
    kind_label,
    plan_token,
)
from .volume import collective_volume_bytes, split_degree


class InfeasibleError(FrameworkError):
    """可行集为空：每个候选方案的解析显存都超过单卡容量。"""


@dataclass(frozen=True)
class SchemeParams:
    """一种切分的参数。流水并行不使用 κ。"""

    kappa: float = 0.0
    tau_s: float = 0.0
    v_s_per_byte: float = 0.0

    def __post_init__(self) -> None:
        if self.kappa < 0.0 or self.tau_s < 0.0 or self.v_s_per_byte < 0.0:
            raise FrameworkError("κ、τ、v 都不能为负")


@dataclass(frozen=True)
class CutGroup:
    kind: str
    degree: int
    n_microbatches: int
    block_t0_s: dict[str, float]
    volume_bytes: float

    @property
    def t0_cut_s(self) -> float:
        return sum(self.block_t0_s.values())


@dataclass(frozen=True)
class Assembly:
    """对照时间拆成未切分的部分，以及每种切分切到的块。"""

    serial_s: float
    block_t0_s: dict[str, float]
    groups: dict[str, CutGroup]

    @property
    def t0_s(self) -> float:
        return sum(self.block_t0_s.values())


@dataclass(frozen=True)
class SchemeTerm:
    kind: str
    degree: int
    t0_cut_s: float
    compute_s: float
    comm_s: float
    volume_bytes: float
    g: float | None


@dataclass(frozen=True)
class Breakdown:
    """一次预测。time_s 是各块时间之和。"""

    time_s: float
    t0_s: float
    serial_s: float
    r: float
    c: float
    g: float | None
    speedup: float
    speedup_upper_bound: float
    mem_bytes: int
    feasible: bool
    block_t0_s: dict[str, float]
    block_time_s: dict[str, float]
    terms: tuple[SchemeTerm, ...]


class TimeModel:
    """拟合得到的预测模型。evaluate 只读它，不更新参数。"""

    def __init__(
        self,
        spec: ModelSpec,
        baselines: dict[str, BlockBaseline],
        schemes: dict[str, SchemeParams],
    ) -> None:
        missing = [block.name for block in spec.blocks if block.name not in baselines]
        if missing:
            raise FrameworkError(f"还没有这些块的基准时间: {', '.join(missing)}")
        self.spec = spec
        self.baselines = dict(baselines)
        self.schemes = dict(schemes)

    def predict(self, load: Load, plan: Plan) -> Breakdown:
        return predict(self, load, plan)

    def select(self, load: Load, plans: list[Plan]) -> Plan:
        """在解析显存不超过单卡容量的方案里，取预测时间最短的。"""
        feasible: list[tuple[float, str, Plan]] = []
        for plan in plans:
            breakdown = self.predict(load, plan)
            if breakdown.feasible:
                feasible.append((breakdown.time_s, plan_token(plan), plan))
        if not feasible:
            raise InfeasibleError("这些方案的解析显存都超过单卡容量")
        feasible.sort(key=lambda item: (item[0], item[1]))
        return feasible[0][2]


def assemble(
    spec: ModelSpec,
    baselines: dict[str, BlockBaseline],
    load: Load,
    plan: Plan,
) -> Assembly:
    ensure_plan(spec, plan)
    serial_s = 0.0
    block_t0_s: dict[str, float] = {}
    buckets: dict[str, list] = {}
    for block in spec.blocks:
        t0_s = baseline_time(block, baselines[block.name], load)
        if t0_s < 0.0:
            raise FrameworkError(f"{block.name} 的基准时间为负")
        block_t0_s[block.name] = t0_s
        split = plan.split_of(block.name)
        if split.kind in ("none", "dp"):
            serial_s += t0_s
            continue
        buckets.setdefault(split.kind, []).append(block)

    groups: dict[str, CutGroup] = {}
    for kind, blocks in buckets.items():
        splits = [plan.split_of(block.name) for block in blocks]
        degree = split_degree(splits)
        micros = {split.n_microbatches for split in splits}
        if len(micros) != 1:
            raise FrameworkError("流水并行的 micro-batch 数在同一次前向里要一致")
        groups[kind] = CutGroup(
            kind=kind,
            degree=degree,
            n_microbatches=micros.pop(),
            block_t0_s={block.name: block_t0_s[block.name] for block in blocks},
            volume_bytes=collective_volume_bytes(kind, blocks, degree, load, spec.elem_bytes),
        )
    return Assembly(serial_s, block_t0_s, groups)


def predict(model: TimeModel, load: Load, plan: Plan) -> Breakdown:
    assembly = assemble(model.spec, model.baselines, load, plan)
    if assembly.t0_s <= 0.0:
        raise FrameworkError("基准时间为 0，无法计算占比和加速比")
    for kind in assembly.groups:
        if kind not in model.schemes:
            raise FrameworkError(
                f"{kind_label(kind)} 还没有参数。"
                "第二阶段只用单独使用该切分的样本来估计，混合样本不参与拟合。"
            )

    terms: list[SchemeTerm] = []
    block_time_s = dict(assembly.block_t0_s)
    for kind, group in assembly.groups.items():
        params = model.schemes[kind]
        compute_s, comm_s, g = _group_time(group, params)
        terms.append(
            SchemeTerm(
                kind=kind,
                degree=group.degree,
                t0_cut_s=group.t0_cut_s,
                compute_s=compute_s,
                comm_s=comm_s,
                volume_bytes=group.volume_bytes,
                g=g,
            )
        )
        _write_block_times(block_time_s, group, compute_s, comm_s, g)

    time_s = assembly.serial_s + sum(term.compute_s + term.comm_s for term in terms)
    t0_s = assembly.t0_s
    t_cut_s = sum(term.t0_cut_s for term in terms)
    comm_s = sum(term.comm_s for term in terms)
    serial_fraction = assembly.serial_s / t0_s
    g_value = terms[0].g if len(terms) == 1 else None
    mem = memory_parts(model.spec, load, plan).total_bytes
    return Breakdown(
        time_s=time_s,
        t0_s=t0_s,
        serial_s=assembly.serial_s,
        r=t_cut_s / t0_s,
        c=comm_s / t0_s,
        g=g_value,
        speedup=t0_s / time_s,
        speedup_upper_bound=speedup_upper_bound(serial_fraction),
        mem_bytes=mem,
        feasible=mem <= model.spec.capacity_bytes,
        block_t0_s=dict(assembly.block_t0_s),
        block_time_s=block_time_s,
        terms=tuple(terms),
    )


def _group_time(group: CutGroup, params: SchemeParams) -> tuple[float, float, float | None]:
    if group.kind == "pp":
        compute_s = group.t0_cut_s * pipeline_factor(group.degree, group.n_microbatches)
        g = None
    else:
        g = compute_speedup(params.kappa, group.degree)
        compute_s = group.t0_cut_s / g
    comm_s = params.tau_s + params.v_s_per_byte * group.volume_bytes
    return compute_s, comm_s, g


def _write_block_times(
    block_time_s: dict[str, float],
    group: CutGroup,
    compute_s: float,
    comm_s: float,
    g: float | None,
) -> None:
    names = list(group.block_t0_s)
    if group.t0_cut_s == 0.0:
        block_time_s[names[0]] = comm_s
        for name in names[1:]:
            block_time_s[name] = 0.0
        return
    for name, t0_s in group.block_t0_s.items():
        if group.kind == "pp":
            compute_i = t0_s * pipeline_factor(group.degree, group.n_microbatches)
        else:
            compute_i = t0_s / g
        block_time_s[name] = compute_i + comm_s * (t0_s / group.t0_cut_s)


def mean_abs_log_error(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    total = 0.0
    for predicted, measured in pairs:
        total += abs(math.log(predicted) - math.log(measured))
    return total / len(pairs)
