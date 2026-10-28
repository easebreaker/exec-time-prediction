"""两阶段拟合。

第一阶段只用对照运行的分段计时，拟合各块基准时间，从而得到 r。
第二阶段每种切分只用单独使用该切分的样本，估计 κ、τ、v。
混合切分不参与拟合，留到测试里检查各块时间能否相加。
损失是端到端时间的相对误差，避免长序列把短序列淹没。
κ 和 v 要分开，需要计算量与通信量随负载的比例不同，或同一种切分有多档并行度。
并行度只有一档、且二者成比例时，预测时间仍可用，但这两个参数的拆分不唯一。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .baseline import features, fit_baselines
from .formula import linear_cut
from .linalg import fit_relative_linear
from .predict import SchemeParams, TimeModel, assemble, mean_abs_log_error, predict
from .schema import FrameworkError, ModelSpec, Sample, ensure_plan, is_control, kind_label
from .schema import active_kinds


@dataclass(frozen=True)
class FitReport:
    model: TimeModel
    n_control: int
    n_control_skipped: int
    n_by_scheme: dict[str, int]
    n_mixed_unused: int
    inactive_parameters: dict[str, tuple[str, ...]]
    baseline_rms_s: dict[str, float]
    train_mean_abs_log_error: float


_PARAM_NAMES = ("kappa", "tau", "v")


def fit(spec: ModelSpec, samples: list[Sample]) -> FitReport:
    controls: list[Sample] = []
    skipped = 0
    by_scheme: dict[str, list[Sample]] = {}
    mixed = 0
    for sample in samples:
        ensure_plan(spec, sample.plan)
        kinds = active_kinds(spec, sample.plan)
        if is_control(spec, sample.plan):
            if _has_full_profile(spec, sample):
                controls.append(sample)
            else:
                skipped += 1
        elif len(kinds) == 1:
            kind = next(iter(kinds))
            by_scheme.setdefault(kind, []).append(sample)
        else:
            mixed += 1
    if not controls:
        raise FrameworkError("需要至少一组带全部分段时间的对照运行，用来拟合各块基准时间")

    rows_by_block = {block.name: [] for block in spec.blocks}
    times_by_block = {block.name: [] for block in spec.blocks}
    for sample in controls:
        for block in spec.blocks:
            rows_by_block[block.name].append(features(block, sample.load))
            times_by_block[block.name].append(sample.block_times_s[block.name])
    baselines = fit_baselines(spec, rows_by_block, times_by_block)
    baseline_rms = _baseline_rms(spec, baselines, controls)

    schemes: dict[str, SchemeParams] = {}
    inactive: dict[str, tuple[str, ...]] = {}
    for kind, group_samples in by_scheme.items():
        params, unused = _fit_scheme(spec, baselines, kind, group_samples)
        schemes[kind] = params
        inactive[kind] = unused

    model = TimeModel(spec, baselines, schemes)
    pairs = []
    for group_samples in by_scheme.values():
        for sample in group_samples:
            predicted = predict(model, sample.load, sample.plan).time_s
            pairs.append((predicted, sample.time_s))
    return FitReport(
        model=model,
        n_control=len(controls),
        n_control_skipped=skipped,
        n_by_scheme={kind: len(group) for kind, group in by_scheme.items()},
        n_mixed_unused=mixed,
        inactive_parameters=inactive,
        baseline_rms_s=baseline_rms,
        train_mean_abs_log_error=mean_abs_log_error(pairs),
    )


def _has_full_profile(spec: ModelSpec, sample: Sample) -> bool:
    if not sample.block_times_s:
        return False
    return all(block.name in sample.block_times_s for block in spec.blocks)


def _baseline_rms(spec: ModelSpec, baselines, controls: list[Sample]) -> dict[str, float]:
    rms: dict[str, float] = {}
    for block in spec.blocks:
        error = 0.0
        for sample in controls:
            predicted = baselines[block.name].time_s(features(block, sample.load))
            gap = predicted - sample.block_times_s[block.name]
            error += gap * gap
        rms[block.name] = math.sqrt(error / len(controls))
    return rms


def _fit_scheme(spec, baselines, kind: str, samples: list[Sample]) -> tuple[SchemeParams, tuple[str, ...]]:
    raw_rows: list[list[float]] = []
    base: list[float] = []
    measured: list[float] = []
    for sample in samples:
        assembly = assemble(spec, baselines, sample.load, sample.plan)
        group = assembly.groups[kind]
        offset_s, feature_s = linear_cut(
            group.t0_cut_s, kind, group.degree, group.n_microbatches
        )
        raw_rows.append([feature_s, 1.0, group.volume_bytes])
        base.append(assembly.serial_s + offset_s)
        measured.append(sample.time_s)
    active = [
        index
        for index in range(3)
        if any(abs(row[index]) > 0.0 for row in raw_rows)
    ]
    if len(samples) < len(active):
        raise FrameworkError(
            f"{kind_label(kind)} 有 {len(samples)} 条单独切分的样本，"
            f"少于要估计的参数个数 {len(active)}"
        )
    rows = [[row[index] for index in active] for row in raw_rows]
    theta = fit_relative_linear(rows, base, measured)
    values = [0.0, 0.0, 0.0]
    for index, value in zip(active, theta):
        values[index] = value
    unused = tuple(_PARAM_NAMES[index] for index in range(3) if index not in active)
    return SchemeParams(values[0], values[1], values[2]), unused
