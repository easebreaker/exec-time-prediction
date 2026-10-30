"""按阶段做的三组实验：复杂度、切分加速比、通信。

复杂度实验只用未切分的运行。一次只动一个轴，从固定的单项里做相对误差的
非负最小二乘。多出来的一项必须把误差明显降下来才保留。误差打平时留下标称
算子的主项，避免和 batch 成比例的 lookup 把 L² 或转置换掉。

切分实验一次只拟合一种切分。κ 和 v 在样本里成比例时标成没有分开，
仍给出一组能对上时间的数，但不把它们当成各自的物理量。
测到了通信时间时，再比较「计算加通信」和「二者取较大值再加没盖住的一段」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .formula import speedup_upper_bound
from .linalg import column_rank, dot, fit_relative_linear, nnls
from .schema import FrameworkError
from .stage import (
    FEATURE_ORDER,
    PRIMARY_TERM,
    SPLIT_LABEL,
    StageLoad,
    StageSpec,
    StageSplit,
    _LOAD_FIELDS,
    compute_parts,
    default_load,
    features,
    scaling_power,
    volume_bytes,
)


@dataclass(frozen=True)
class StageObservation:
    load: StageLoad
    split: StageSplit
    time_s: float
    comm_bytes: float | None = None
    comm_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.time_s <= 0.0:
            raise FrameworkError("实验时间要为正")
        if self.comm_bytes is not None and self.comm_bytes < 0.0:
            raise FrameworkError("通信字节数不能为负")
        if self.comm_time_s is not None and self.comm_time_s < 0.0:
            raise FrameworkError("通信时间不能为负")


@dataclass(frozen=True)
class ComplexityReport:
    stage: str
    operator: str
    terms: tuple[tuple[str, float], ...]
    mape: float
    identified: bool
    matches_operator: bool
    depends_on_length: bool
    length_checked: bool
    width_log_slope: float | None
    reason: str

    def coefficients(self) -> dict[str, float]:
        return dict(self.terms)


@dataclass(frozen=True)
class SplitReport:
    pattern: str
    kappa: float
    tau_s: float
    v_s_per_byte: float
    uncovered_s: float
    overlap: bool
    identified: bool
    inactive: tuple[str, ...]
    train_mape: float
    volume_matches: bool | None
    reason: str
    # 样本里只有一个并行度时记下来，预测可以不再另传 plan。多个并行度时为空。
    degree: int | None = None
    n_microbatches: int | None = None


@dataclass(frozen=True)
class StageEstimate:
    time_s: float
    compute_s: float
    comm_s: float
    unsplit_s: float
    speedup: float
    speedup_upper_bound: float
    identified: bool
    overlap: bool
    split: StageSplit


@dataclass(frozen=True)
class NetworkEstimate:
    time_s: float
    stages: tuple[tuple[str, StageEstimate], ...]


def stage_time(
    spec: StageSpec,
    coefficients: dict[str, float],
    load: StageLoad,
    split: StageSplit,
    *,
    kappa: float = 0.0,
    tau_s: float = 0.0,
    v_s_per_byte: float = 0.0,
    overlap: bool = False,
    uncovered_s: float = 0.0,
) -> tuple[float, float, float]:
    """返回总时间、计算时间、通信时间。"""
    if tau_s < 0.0 or v_s_per_byte < 0.0 or uncovered_s < 0.0:
        raise FrameworkError("τ、v 和未掩盖时间都不能为负")
    compute, _, _ = compute_parts(spec, coefficients, load, split, kappa)
    if split.kind == "none":
        if compute <= 0.0:
            raise FrameworkError(f"{spec.name} 的预测时间为 0")
        return compute, compute, 0.0
    comm = tau_s + v_s_per_byte * volume_bytes(spec, load, split)
    total = max(compute, comm) + uncovered_s if overlap else compute + comm
    if total <= 0.0:
        raise FrameworkError(f"{spec.name} 的预测时间为 0")
    return total, compute, comm


_BOUND_GRIDS = {
    "length": (128, 256, 512, 1024, 2048, 4096),
    "tokens": (4, 8, 16, 32),
    "groups": (1, 2, 4, 8),
    "n_query": (1, 2, 4, 8, 16),
    "length_kv": (128, 512, 2048),
    "n_id": (16, 64, 256, 1024),
    "n_head": (1, 2, 4, 8),
    "width": (32, 64, 128, 256),
}


def complexity_grid(spec: StageSpec, base: StageLoad | None = None) -> list[StageLoad]:
    """每个轴单独取一串倍数。长度轴始终在里面，用来判断该阶段读不读用户序列 L。

    接到 extras 上的轴会补上默认值并单独扫描。否则 boost 的 token 数不随
    tokens 列变化，转置项和 lookup 在样本里分不开。
    """
    base = _seed_bound_axes(spec, base or default_load())
    loads = [base]
    for batch in (1, 2, 4, 8):
        _push(loads, base.varied(batch=batch))
    for length in (128, 256, 512, 1024, 2048, 4096):
        _push(loads, base.varied(length=length))
    for tokens in (4, 8, 16, 32):
        _push(loads, base.varied(tokens=tokens))
    if not spec.groups_follow_length:
        for groups in (1, 2, 4, 8):
            _push(loads, base.varied(groups=groups))
    for n_query in (1, 2, 4, 8, 16):
        _push(loads, base.varied(n_query=n_query))
    for length_kv in (128, 512, 2048):
        _push(loads, base.varied(length_kv=length_kv))
    for n_id in (16, 64, 256, 1024):
        _push(loads, base.varied(n_id=n_id))
    for n_head in (1, 2, 4, 8):
        _push(loads, base.varied(n_head=n_head))
    for width in (32, 64, 128, 256):
        ratio = width / base.width
        _push(
            loads,
            base.varied(
                width=width,
                hidden=max(1, round(base.hidden * ratio)),
                d_head=max(1, round(base.d_head * ratio)),
            ),
        )
    _push(loads, base.varied(n_real=max(1, base.n_real // 4)))
    for key, value in base.extras.items():
        for factor in (2, 4, 8):
            _push(loads, base.varied(extras={key: value * factor}))
    _sweep_bound_axes(spec, base, loads)
    return loads


def fit_complexity(spec: StageSpec, observations: list[StageObservation]) -> ComplexityReport:
    if any(obs.split.kind != "none" for obs in observations):
        raise FrameworkError("复杂度实验只用未切分的运行")
    if len(observations) < 2:
        raise FrameworkError("复杂度实验至少需要两次未切分运行")
    loads = [obs.load for obs in observations]
    times = [obs.time_s for obs in observations]
    eligible = _eligible_terms(spec, loads)
    fitted: list[tuple[dict[str, float], float]] = []
    for mask in range(1 << len(eligible)):
        if bin(mask).count("1") > 3:
            continue
        names = [eligible[index] for index in range(len(eligible)) if mask & (1 << index)]
        coefficients, mape = _fit_terms(spec, loads, times, names)
        if math.isfinite(mape) and coefficients:
            fitted.append((coefficients, mape))
    if not fitted:
        raise FrameworkError(f"{spec.name} 的复杂度没有可用的拟合")
    # 误差打平时保留标称主项。与 batch 成比例时，按名字会先选到
    # cross_attn、lookup，长序列上的 L² 或 boost 的转置就被换成常数。
    primary = PRIMARY_TERM[spec.operator]
    best_mape = min(mape for _, mape in fitted)
    declared = [item for item in fitted if primary in item[0] and item[1] <= best_mape + 1e-4]
    if declared:
        pool = declared
    else:
        tolerance = best_mape + max(0.05 * best_mape, 1e-4)
        pool = [item for item in fitted if item[1] <= tolerance]
    coefficients, mape = min(pool, key=lambda item: (len(item[0]), item[1], tuple(item[0])))
    rows = [[features(spec, load)[name] for name in coefficients] for load in loads] if coefficients else []
    identified = bool(coefficients) and column_rank(rows) == len(coefficients)
    matches = PRIMARY_TERM[spec.operator] in coefficients
    length_checked, depends = _length_effect(observations)
    slope = _width_slope(observations)
    if not coefficients:
        reason = "没有识别出随负载变化的项"
    elif not matches:
        reason = "选中的项与标称算子不一致"
    elif not identified:
        reason = "选中的项在这批样本里成比例，没有分开"
    else:
        reason = ""
    terms = tuple((name, coefficients[name]) for name in FEATURE_ORDER if name in coefficients)
    return ComplexityReport(
        stage=spec.name,
        operator=spec.operator,
        terms=terms,
        mape=mape,
        identified=identified,
        matches_operator=matches,
        depends_on_length=depends,
        length_checked=length_checked,
        width_log_slope=slope,
        reason=reason,
    )


def fit_split(
    spec: StageSpec,
    complexity: ComplexityReport,
    observations: list[StageObservation],
) -> SplitReport:
    if not observations or any(obs.split.kind == "none" for obs in observations):
        raise FrameworkError("切分实验只用切分后的运行")
    patterns = {obs.split.kind for obs in observations}
    if len(patterns) != 1:
        raise FrameworkError("一次只拟合一种切分")
    pattern = patterns.pop()
    for obs in observations:
        if obs.split.kind != pattern:
            raise FrameworkError("一次只拟合一种切分")
        # 不合法的切分在算体积时抛出。
        volume_bytes(spec, obs.load, obs.split)
    coefficients = complexity.coefficients()
    kappa_column = []
    offsets = []
    volumes = []
    measured = []
    for obs in observations:
        _, kappa_feature, offset = compute_parts(spec, coefficients, obs.load, obs.split, 0.0)
        kappa_column.append(kappa_feature)
        offsets.append(offset)
        volumes.append(volume_bytes(spec, obs.load, obs.split))
        measured.append(obs.time_s)
    names, rows = _split_columns(kappa_column, volumes)
    if len(observations) < len(names):
        theta = [0.0] * len(names)
        identified = False
        reason = "样本数少于要分开的参数个数"
        train_mape = 1.0
    else:
        rank = column_rank(rows)
        identified = rank == len(names)
        theta = fit_relative_linear(rows, offsets, measured)
        reason = "" if identified else "κ 与 v 在这批样本里没有分开"
        preds = [offsets[i] + dot(rows[i], theta) for i in range(len(measured))]
        train_mape = _mape(preds, measured)
    values = {name: value for name, value in zip(names, theta)}
    median_time = sorted(measured)[len(measured) // 2]
    inactive: list[str] = []
    for name in ("kappa", "tau", "v"):
        if name not in values:
            inactive.append(name)
            continue
        column = {"kappa": kappa_column, "tau": [1.0] * len(measured), "v": volumes}[name]
        peak = max((abs(value) for value in column), default=1.0) or 1.0
        if values[name] * peak < 1e-5 * median_time:
            values[name] = 0.0
            inactive.append(name)
    kappa = values.get("kappa", 0.0)
    tau = values.get("tau", 0.0)
    v_byte = values.get("v", 0.0)
    volume_matches, _volume_mape = _volume_agreement(spec, observations)
    overlap = False
    uncovered = 0.0
    if all(obs.comm_time_s is not None for obs in observations):
        overlap, uncovered, kappa, tau, v_byte, train_mape = _choose_overlap(
            spec,
            coefficients,
            observations,
            kappa,
            tau,
            v_byte,
            train_mape,
        )
    if volume_matches is False:
        identified = False
        reason = "通信体积和公式不一致"
    degrees = {obs.split.degree for obs in observations}
    micros = {obs.split.n_microbatches for obs in observations}
    return SplitReport(
        pattern=pattern,
        kappa=kappa,
        tau_s=tau,
        v_s_per_byte=v_byte,
        uncovered_s=uncovered,
        overlap=overlap,
        identified=identified,
        inactive=tuple(inactive),
        train_mape=train_mape,
        volume_matches=volume_matches,
        reason=reason,
        degree=degrees.pop() if len(degrees) == 1 else None,
        n_microbatches=micros.pop() if len(micros) == 1 else None,
    )


def predict_stage(
    spec: StageSpec,
    complexity: ComplexityReport,
    load: StageLoad,
    split: StageSplit,
    split_fit: SplitReport | None = None,
) -> StageEstimate:
    coefficients = complexity.coefficients()
    unsplit, _, _ = stage_time(spec, coefficients, load, StageSplit("none"))
    if split.kind == "none":
        return StageEstimate(unsplit, unsplit, 0.0, unsplit, 1.0, 1.0, True, False, split)
    if split_fit is None or split_fit.pattern != split.kind:
        raise FrameworkError(f"{spec.name} 还没有 {SPLIT_LABEL[split.kind]} 的参数")
    total, compute, comm = stage_time(
        spec,
        coefficients,
        load,
        split,
        kappa=split_fit.kappa,
        tau_s=split_fit.tau_s,
        v_s_per_byte=split_fit.v_s_per_byte,
        overlap=split_fit.overlap,
        uncovered_s=split_fit.uncovered_s,
    )
    serial = _serial_compute(spec, coefficients, load, split)
    if split.kind == "pp":
        upper = float(split.degree)
    else:
        upper = speedup_upper_bound(serial / unsplit)
    return StageEstimate(
        time_s=total,
        compute_s=compute,
        comm_s=comm,
        unsplit_s=unsplit,
        speedup=unsplit / total,
        speedup_upper_bound=upper,
        identified=split_fit.identified and complexity.identified,
        overlap=split_fit.overlap,
        split=split,
    )


def predict_network(
    stages: tuple[StageSpec, ...],
    complexities: dict[str, ComplexityReport],
    split_fits: dict[str, SplitReport],
    load: StageLoad,
    plan: dict[str, StageSplit] | None = None,
) -> NetworkEstimate:
    """按阶段相加。plan 里写了具体切分就用它；缺省或写了不切分时，用该阶段的拟合。

    拟合只含一个并行度时，degree 来自样本。含多个并行度而又没在 plan 里
    写明时直接报错，不再把切分悄悄当成 none。
    """
    plan = plan or {}
    pieces: list[tuple[str, StageEstimate]] = []
    for spec in stages:
        if spec.name not in complexities:
            raise FrameworkError(f"{spec.name} 还没有复杂度")
        fit = split_fits.get(spec.name)
        planned = plan.get(spec.name)
        if fit is not None and (planned is None or planned.kind == "none"):
            split = _split_from_report(spec, fit)
        elif planned is not None:
            split = planned
        else:
            split = StageSplit("none")
        pieces.append((spec.name, predict_stage(spec, complexities[spec.name], load, split, fit)))
    return NetworkEstimate(sum(piece.time_s for _, piece in pieces), tuple(pieces))


def _seed_bound_axes(spec: StageSpec, base: StageLoad) -> StageLoad:
    extras = {}
    for role, field in spec.axes.items():
        if field in _LOAD_FIELDS or field in base.extras or role not in _LOAD_FIELDS:
            continue
        extras[field] = int(getattr(base, role))
    if not extras:
        return base
    return base.varied(extras=extras)


def _sweep_bound_axes(spec: StageSpec, base: StageLoad, loads: list[StageLoad]) -> None:
    for role, field in spec.axes.items():
        if field in _LOAD_FIELDS or (role == "groups" and spec.groups_follow_length):
            continue
        for value in _BOUND_GRIDS.get(role, ()):
            _push(loads, base.varied(extras={field: int(value)}))


def _split_from_report(spec: StageSpec, fit: SplitReport) -> StageSplit:
    if fit.pattern == "none":
        return StageSplit("none")
    if fit.degree is None or fit.n_microbatches is None:
        label = SPLIT_LABEL[fit.pattern]
        raise FrameworkError(
            f"{spec.name} 的{label}拟合了多个并行度，预测时要在 plan 里写明切分"
        )
    return StageSplit(fit.pattern, fit.degree, fit.n_microbatches)


def _push(loads: list[StageLoad], load: StageLoad) -> None:
    if load not in loads:
        loads.append(load)


def _varies(column: list[float]) -> bool:
    if not column:
        return False
    span = max(column) - min(column)
    scale = max(1.0, max(abs(value) for value in column))
    return span > 1e-9 * scale


def _mape(predicted: list[float], measured: list[float]) -> float:
    if not measured:
        raise FrameworkError("没有样本")
    total = 0.0
    for pred, truth in zip(predicted, measured):
        if truth <= 0.0:
            raise FrameworkError("实验时间要为正")
        # 单点相对误差封顶，否则一条极短样本的误差能到 1e6，
        # 预测成 0 的空模型（误差恰好为 1）会反而胜出。
        total += min(abs(pred - truth) / truth, 10.0)
    return total / len(measured)


def _eligible_terms(spec: StageSpec, loads: list[StageLoad]) -> list[str]:
    eligible = []
    for name in FEATURE_ORDER:
        column = [features(spec, load)[name] for load in loads]
        if name == "intercept" or _varies(column):
            eligible.append(name)
    return eligible


def _fit_terms(
    spec: StageSpec,
    loads: list[StageLoad],
    times: list[float],
    names: list[str],
) -> tuple[dict[str, float], float]:
    if not names:
        return {}, _mape([0.0] * len(times), times)
    rows = [[features(spec, load)[name] for name in names] for load in loads]
    if column_rank(rows) < len(names):
        return {}, math.inf
    theta = fit_relative_linear(rows, [0.0] * len(times), times)
    median_time = sorted(times)[len(times) // 2]
    primary = PRIMARY_TERM[spec.operator]
    kept: list[str] = []
    for index, name in enumerate(names):
        column = [row[index] for row in rows]
        peak = max(abs(value) for value in column) or 1.0
        # 主项的列范数比截距大很多个数量级，系数本身会很小。
        # 用 coef×峰值 和 1e-5×中位时间比较时，主项会被当成 0 丢掉，
        # 阶段时间只剩常数，更长的负载上整网会偏低好几倍。
        if name == primary and theta[index] > 0.0:
            kept.append(name)
            continue
        if theta[index] * peak >= 1e-5 * median_time:
            kept.append(name)
    if len(kept) < len(names):
        return _fit_terms(spec, loads, times, kept)
    coefficients = {name: value for name, value in zip(names, theta)}
    predicted = [dot(row, theta) for row in rows]
    return coefficients, _mape(predicted, times)


def _dense(load: StageLoad) -> bool:
    return load.n_real == load.batch * load.length


def _except_length(load: StageLoad) -> tuple:
    return (
        load.batch,
        load.n_id,
        load.n_query,
        load.length_kv,
        load.tokens,
        load.groups,
        load.width,
        load.hidden,
        load.n_head,
        load.d_head,
        tuple(sorted(load.extras.items())),
    )


def _except_width(load: StageLoad) -> tuple:
    return (
        load.batch,
        load.length,
        load.n_real,
        load.n_id,
        load.n_query,
        load.length_kv,
        load.tokens,
        load.groups,
        load.n_head,
        tuple(sorted(load.extras.items())),
    )


def _length_effect(observations: list[StageObservation]) -> tuple[bool, bool]:
    groups: dict[tuple, list[StageObservation]] = {}
    for obs in observations:
        if _dense(obs.load):
            groups.setdefault(_except_length(obs.load), []).append(obs)
    checked = False
    depends = False
    for group in groups.values():
        by_length: dict[int, list[float]] = {}
        for obs in group:
            by_length.setdefault(obs.load.length, []).append(obs.time_s)
        if len(by_length) < 2:
            continue
        checked = True
        means = [sum(values) / len(values) for values in by_length.values()]
        if max(means) / min(means) > 1.05:
            depends = True
    return checked, depends


def _width_slope(observations: list[StageObservation]) -> float | None:
    groups: dict[tuple, list[StageObservation]] = {}
    for obs in observations:
        groups.setdefault(_except_width(obs.load), []).append(obs)
    best: list[StageObservation] | None = None
    best_widths = 0
    for group in groups.values():
        widths = {obs.load.width for obs in group}
        if len(widths) > best_widths:
            best = group
            best_widths = len(widths)
    if best is None or best_widths < 2:
        return None
    by_width: dict[int, list[float]] = {}
    for obs in best:
        by_width.setdefault(obs.load.width, []).append(obs.time_s)
    widths = sorted(by_width)
    means = [sum(by_width[width]) / len(by_width[width]) for width in widths]
    if min(means) <= 0.0 or max(means) / min(means) <= 1.01:
        return None
    log_x = [math.log(width) for width in widths]
    log_y = [math.log(value) for value in means]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    variance = sum((value - mean_x) ** 2 for value in log_x)
    if variance == 0.0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_x, log_y))
    return covariance / variance


def _split_columns(
    kappa_column: list[float],
    volumes: list[float],
) -> tuple[list[str], list[list[float]]]:
    names: list[str] = []
    columns: list[list[float]] = []
    if any(abs(value) > 0.0 for value in kappa_column):
        names.append("kappa")
        columns.append(kappa_column)
    names.append("tau")
    columns.append([1.0] * len(kappa_column))
    if any(abs(value) > 0.0 for value in volumes):
        names.append("v")
        columns.append(volumes)
    rows = [[column[row] for column in columns] for row in range(len(kappa_column))]
    return names, rows


def _volume_agreement(
    spec: StageSpec,
    observations: list[StageObservation],
) -> tuple[bool | None, float | None]:
    if any(obs.comm_bytes is None for obs in observations):
        return None, None
    errors = []
    for obs in observations:
        formula = volume_bytes(spec, obs.load, obs.split)
        measured = obs.comm_bytes or 0.0
        if measured == 0.0 and formula == 0.0:
            errors.append(0.0)
        else:
            errors.append(abs(formula - measured) / max(measured, formula, 1.0))
    mape = sum(errors) / len(errors)
    return mape <= 0.05, mape


def _choose_overlap(
    spec: StageSpec,
    coefficients: dict[str, float],
    observations: list[StageObservation],
    kappa: float,
    tau: float,
    v_byte: float,
    additive_mape: float,
) -> tuple[bool, float, float, float, float, float]:
    comm_rows = []
    comm_times = []
    for obs in observations:
        comm_rows.append([1.0, volume_bytes(spec, obs.load, obs.split)])
        comm_times.append(obs.comm_time_s or 0.0)
    if column_rank(comm_rows) == 2:
        comm_tau, comm_v = fit_relative_linear(comm_rows, [0.0] * len(comm_times), comm_times)
    else:
        comm_tau, comm_v = tau, v_byte
    additive_error = _pair_mape(
        spec, coefficients, observations, kappa, measured_comm=True, overlap=False, uncovered=0.0
    )
    best_kappa = kappa
    best_uncovered = 0.0
    best_error = math.inf
    for step in range(0, 201):
        trial = step / 100.0
        computes = []
        residuals = []
        for obs in observations:
            compute, _, _ = compute_parts(spec, coefficients, obs.load, obs.split, trial)
            comm = obs.comm_time_s or 0.0
            computes.append(max(compute, comm))
            residuals.append(obs.time_s - max(compute, comm))
        uncovered = nnls([[1.0] for _ in residuals], residuals)[0]
        predicted = [value + uncovered for value in computes]
        error = _mape(predicted, [obs.time_s for obs in observations])
        if error < best_error:
            best_error = error
            best_kappa = trial
            best_uncovered = uncovered
    if best_error + 1e-6 < additive_error - max(0.02, 0.05 * additive_error):
        return True, best_uncovered, best_kappa, max(0.0, comm_tau), max(0.0, comm_v), best_error
    return False, 0.0, kappa, tau, v_byte, additive_mape


def _pair_mape(
    spec: StageSpec,
    coefficients: dict[str, float],
    observations: list[StageObservation],
    kappa: float,
    *,
    measured_comm: bool,
    overlap: bool,
    uncovered: float,
) -> float:
    predicted = []
    for obs in observations:
        compute, _, _ = compute_parts(spec, coefficients, obs.load, obs.split, kappa)
        comm = obs.comm_time_s if measured_comm else 0.0
        comm = comm or 0.0
        predicted.append(max(compute, comm) + uncovered if overlap else compute + comm)
    return _mape(predicted, [obs.time_s for obs in observations])


def _serial_compute(
    spec: StageSpec,
    coefficients: dict[str, float],
    load: StageLoad,
    split: StageSplit,
) -> float:
    feats = features(spec, load)
    serial = 0.0
    for name in FEATURE_ORDER:
        if split.kind != "pp" and scaling_power(split.kind, name) == 0:
            serial += coefficients.get(name, 0.0) * feats[name]
    return serial
