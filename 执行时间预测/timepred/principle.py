"""原理性测定：生成测量表、按判据定位真实复杂度。

配套文档 `复杂度测定-原理方案.md`。两组功能：

1. `experiment_plan()` —— 六个实验的请求清单，直接写成阶段测量表的模板。
2. `identify_*()` —— 每个实验一个判定函数，输入测量点，输出结论与置信度。

判定函数刻意只做「读数据 → 给结论」，不做模型选择：结论是
「台阶存在 / 不存在」「线性 / 阶梯」「L* 在哪」这类可证伪的陈述，
再由 `verdict()` 汇总成对复杂度形式的结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .schema import FrameworkError
from .stage import StageLoad, StageSplit

# ---------------------------------------------------------------- 实验清单


@dataclass(frozen=True)
class Trial:
    """一次测量请求。id 同时是测量表的行名。"""

    experiment: str
    id: str
    load: StageLoad
    split: StageSplit
    note: str


_LADDER = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
_OFFGRID = (8192, 8320, 8448)
_OFFGRID_Q = (8192, 8256, 8320)
_BH = (1, 2, 4, 8, 16, 32)
_N_DEGREES = (1, 2, 4, 8)


def experiment_plan(layers: int = 1) -> tuple[Trial, ...]:
    """六个实验的完整清单，共 36 条（不含 P1 的重复次数）。"""
    trials: list[Trial] = []

    # E1a：只改层数，看 b 是否 ∝ kernel 数
    for n in (1, 2, 4):
        trials.append(
            Trial(
                "E1a",
                f"E1a-L{layers}-N{n}",
                StageLoad(batch=1, length=4096, n_head=8, d_head=64),
                StageSplit("none"),
                f"注意力层数 {n}；只改 kernel 数，L 与 N 不变",
            )
        )

    # E1c：固定 L 与层数，扫并行度（含 N=1）
    for degree in _N_DEGREES:
        split = StageSplit("none") if degree == 1 else StageSplit("cp_q", degree)
        trials.append(
            Trial(
                "E1c",
                f"E1c-N{degree}",
                StageLoad(batch=1, length=4096, n_head=8, d_head=64),
                split,
                f"cp_q 并行度 {degree}；N=1 是现有数据缺的那一档",
            )
        )

    # E2：两个 GEMM 分开，只改 d_head 与 d_v
    for head, value in ((64, 64), (128, 64), (64, 128)):
        trials.append(
            Trial(
                "E2",
                f"E2-D{head}-V{value}",
                StageLoad(batch=1, length=4096, n_head=8, d_head=head),
                StageSplit("none"),
                f"d_head={head}, d_v={value}；固定 B·H·L²，只改两个 GEMM 的相对权重",
            )
        )

    # E3：非格点，让 ceil 的台阶显形。cp8 的跳幅最大（+14.3%），cp2 只有 +4.7% 不可测。
    for length in _OFFGRID:
        trials.append(
            Trial(
                "E3a",
                f"E3a-L{length}",
                StageLoad(batch=1, length=length, n_head=8, d_head=64),
                StageSplit("cp_q", 8),
                "cp8：块数 512/585/594，首对跳 +14.3%；L 不是 128 的倍数",
            )
        )
    for length in _OFFGRID_Q:
        trials.append(
            Trial(
                "E3b",
                f"E3b-L{length}",
                StageLoad(batch=1, length=length, n_head=8, d_head=64),
                StageSplit("cp_q", 2),
                "cp2：Q 侧块数 32/33/34；只在 Q 侧跨格，用于区分两侧基本块",
            )
        )

    # E4：扫 B·H，看外循环是否并行于核
    for bh in _BH:
        trials.append(
            Trial(
                "E4",
                f"E4-BH{bh:02d}",
                StageLoad(batch=1, length=4096, n_head=bh, d_head=64),
                StageSplit("cp_q", 2),
                f"B·H={bh}；L 与 N 固定，只改可并行的 tile 数",
            )
        )

    # E5：N=1、单层，扫到 65536，定 L*
    for length in _LADDER:
        trials.append(
            Trial(
                "E5",
                f"E5-L{length}",
                StageLoad(batch=1, length=length, n_head=8, d_head=64),
                StageSplit("none"),
                "N=1 单层；对数-对数求局部斜率，过渡中点即 L*",
            )
        )
    return tuple(trials)


def by_experiment(trials: tuple[Trial, ...]) -> dict[str, tuple[Trial, ...]]:
    grouped: dict[str, list[Trial]] = {}
    for trial in trials:
        grouped.setdefault(trial.experiment, []).append(trial)
    return {name: tuple(items) for name, items in grouped.items()}


# ---------------------------------------------------------------- 判据工具


@dataclass(frozen=True)
class Reading:
    """一个测点的读数。`scatter` 是同点重复的 IQR，用于分开信号与噪声。"""

    x: float
    time_s: float
    scatter_s: float = 0.0


def _step_tolerance(readings: list[Reading]) -> float:
    values = [r.scatter_s for r in readings if r.scatter_s > 0]
    return max(values) if values else 0.0


@dataclass(frozen=True)
class Verdict:
    experiment: str
    question: str
    answer: str
    confidence: str
    detail: tuple[str, ...] = field(default=())

    def line(self) -> str:
        return f"[{self.experiment}] {self.question}\n    结论：{self.answer}（{self.confidence}）"


def _fmt(value: float, unit: str = "") -> str:
    return f"{value:.3g}{unit}"


# ---------------------------------------------------------------- 各实验判定


def identify_e1a(layers_pairs: list[tuple[int, float]]) -> Verdict:
    """b 是否 ∝ kernel 数。`layers_pairs` 为 [(层数, 拟合或实围出的 b)]。"""
    if len(layers_pairs) < 2:
        return Verdict("E1a", "b 是否 ∝ kernel 数", "样本不足", "无法判定")
    xs = [float(n) for n, _ in layers_pairs]
    ys = [float(b) for _, b in layers_pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return Verdict("E1a", "b 是否 ∝ kernel 数", "层数取值重复", "无法判定")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    detail = (f"斜率 {_fmt(slope * 1e3, ' ms/层')}", f"截距 {_fmt(intercept * 1e3, ' ms')}", f"R² = {r2:.4f}")
    if r2 > 0.99 and abs(intercept) < 0.1 * abs(slope):
        return Verdict("E1a", "b 是否 ∝ kernel 数", "是，b 是逐 kernel 下发项", "高", detail)
    if r2 > 0.99:
        return Verdict("E1a", "b 是否 ∝ kernel 数", "近似线性，但有与层数无关的常数项", "中", detail)
    return Verdict("E1a", "b 是否 ∝ kernel 数", "不是线性，b 里还有别的成分", "中", detail)


def identify_e1c(readings: list[Reading]) -> Verdict:
    """b 随并行度是否呈阶梯（通信轮次）。"""
    if len(readings) < 3:
        return Verdict("E1c", "b 的增量是否呈轮次阶梯", "样本不足", "无法判定")
    ordered = sorted(readings, key=lambda r: r.x)
    increments = [b.time_s - a.time_s for a, b in zip(ordered, ordered[1:])]
    detail = tuple(
        f"N={ordered[i].x:.0f}→{ordered[i + 1].x:.0f}: b 增 {inc * 1e3:+.2f} ms"
        for i, inc in enumerate(increments)
    )
    tol = _step_tolerance(ordered)
    if tol == 0.0:
        return Verdict("E1c", "b 的增量是否呈轮次阶梯", f"增量 {[round(v * 1e3, 2) for v in increments]} ms", "低（未记抖动）", detail)
    flat = max(increments) - min(increments)
    if flat <= 3 * tol:
        return Verdict("E1c", "b 的增量是否呈轮次阶梯", "增量近似相同，与轮次无关", "中", detail)
    return Verdict("E1c", "b 的增量是否呈轮次阶梯", "增量不等，符合轮次/建链随 N 变", "中", detail)


def identify_e2(triples: list[tuple[int, int, float]]) -> Verdict:
    """两个 GEMM 是否等权。`triples` 为 [(d_head, d_v, time)]。"""
    lookup = {(h, v): t for h, v, t in triples}
    if (64, 64) not in lookup:
        return Verdict("E2", "L² 项宽度是 d_head 还是 d_head+d_v", "缺少基准点 (64,64)", "无法判定")
    base = lookup[(64, 64)]
    ratios = {key: lookup[key] / base for key in ((128, 64), (64, 128)) if key in lookup}
    if len(ratios) < 2:
        return Verdict("E2", "L² 项宽度是 d_head 还是 d_head+d_v", "缺少 (128,64) 或 (64,128)", "无法判定")
    wide_head = ratios[(128, 64)]
    wide_value = ratios[(64, 128)]
    detail = (f"T(128,64)/T(64,64) = {wide_head:.3f}", f"T(64,128)/T(64,64) = {wide_value:.3f}")
    both = 1.5
    tol = 0.15
    if abs(wide_head - both) < tol and abs(wide_value - both) < tol:
        return Verdict("E2", "L² 项宽度", "两个 GEMM 等权，宽度 = d_head + d_v", "高", detail)
    if abs(wide_head - 1.0) < tol and abs(wide_value - 1.0) < tol:
        return Verdict("E2", "L² 项宽度", "与头维无关 ⇒ 主项不是 GEMM", "高", detail)
    return Verdict("E2", "L² 项宽度", "比值不匹配任一预期，需查第三个成分", "中", detail)


def identify_e3(readings: list[Reading], predicted_steps: list[int]) -> Verdict:
    """组内是否阶梯。`predicted_steps` 为每点的块数。

    逐对比较「实测时间比」与「块数比」。每一对都先算信噪比：
    信号 = |块数比 − 1| × 前一点时间，噪声 = 2σ。S/N < 3 的对不参与判定
    —— 否则会把噪声读成台阶（或反过来）。
    """
    if len(readings) < 3:
        return Verdict("E3", "块数台阶是否可见", "样本不足", "无法判定")
    order = sorted(range(len(readings)), key=lambda i: readings[i].x)
    detail: list[str] = []
    usable = 0
    violations = 0
    weak = 0
    for a, b in zip(order, order[1:]):
        ratio_time = readings[b].time_s / readings[a].time_s
        ratio_step = predicted_steps[b] / predicted_steps[a]
        signal = abs(ratio_step - 1.0) * readings[a].time_s
        noise = 2.0 * max(readings[a].scatter_s, readings[b].scatter_s)
        snr = signal / noise if noise > 0 else float("inf")
        agree = abs(ratio_time - ratio_step) * readings[a].time_s <= noise
        tag = "一致" if agree else "冲突"
        detail.append(
            f"L={readings[a].x:.0f}→{readings[b].x:.0f}: 块数比 {ratio_step:.3f} 实测比 {ratio_time:.3f} "
            f"信号 {signal * 1e3:.3f} ms 噪声 {noise * 1e3:.3f} ms S/N {snr:.1f} → {tag}"
        )
        if snr < 3.0:
            weak += 1
            continue
        usable += 1
        if not agree:
            violations += 1
    if usable == 0:
        return Verdict(
            "E3",
            "块数台阶是否可见",
            "全部相邻对的 S/N < 3 ⇒ 台阶不可测，需先降低同点噪声（P1）",
            "高",
            tuple(detail),
        )
    if violations == 0:
        return Verdict(
            "E3",
            "块数台阶是否可见",
            f"可判的 {usable} 对全部一致 ⇒ 台阶成立，ceil 应进模型",
            "高",
            tuple(detail),
        )
    return Verdict(
        "E3",
        "块数台阶是否可见",
        f"{violations}/{usable} 对与台阶预测冲突 ⇒ 台阶不成立（另有 {weak} 对 S/N 不足）",
        "高",
        tuple(detail),
    )


def identify_e4(readings: list[Reading]) -> Verdict:
    """B·H 是线性还是阶梯（不足一波）。"""
    if len(readings) < 3:
        return Verdict("E4", "并行结构：全 tile 并行还是外循环并行", "样本不足", "无法判定")
    ordered = sorted(readings, key=lambda r: r.x)
    base = ordered[0]
    detail = tuple(f"B·H={r.x:.0f}: {r.time_s * 1e3:7.3f} ms" for r in ordered)

    def flat_against_base(r: Reading) -> bool:
        # 相对判据：差值若小于「3 倍相对抖动」或 5%，算同一水平。
        relative_scatter = max(r.scatter_s / r.time_s, base.scatter_s / base.time_s)
        return abs(r.time_s - base.time_s) / base.time_s <= max(3.0 * relative_scatter, 0.05)

    plateau = [r for r in ordered if flat_against_base(r)]
    # 平台要求：至少两个点在同一水平，且后面确实涨起来了。
    if len(plateau) >= 2 and plateau[-1].x < ordered[-1].x:
        return Verdict(
            "E4",
            "并行结构",
            f"B·H ≤ {plateau[-1].x:.0f} 时平坦 ⇒ 存在不足一波的平台",
            "中",
            detail,
        )

    xs = [r.x for r in ordered]
    ys = [r.time_s for r in ordered]
    fit = ys[-1] / xs[-1]
    linear_err = max(abs(y - fit * x) / y for x, y in zip(xs, ys))
    if linear_err < 0.15:
        return Verdict(
            "E4",
            "并行结构",
            "随 B·H 线性 ⇒ 全部 tile 一起分核",
            "中",
            detail + (f"最大相对偏离 {linear_err:.1%}",),
        )
    return Verdict(
        "E4",
        "并行结构",
        "既非线性也非平台 ⇒ 阶梯，台阶在 C·k（见 detail）",
        "中",
        detail,
    )


def identify_e5(readings: list[Reading]) -> Verdict:
    """对数-对数求局部斜率，找 1→2 的过渡中点。

    只在「低端确实是 1 次」时才谈过渡。若第一段斜率就已经 ≥1.5，
    说明整个扫描范围都在计算 bound，不存在过渡。
    """
    if len(readings) < 4:
        return Verdict("E5", "regime 切换点 L*", "样本不足", "无法判定")
    ordered = sorted(readings, key=lambda r: r.x)
    slopes = []
    for a, b in zip(ordered, ordered[1:]):
        slope = math.log(b.time_s / a.time_s) / math.log(b.x / a.x)
        mid = math.sqrt(a.x * b.x)
        slopes.append((mid, slope))
    detail = tuple(f"L≈{mid:7.0f}: 局部斜率 {s:.2f}" for mid, s in slopes)

    if slopes[0][1] >= 1.5:
        return Verdict(
            "E5",
            "regime 切换点 L*",
            f"整个扫描范围内斜率已 ≥{slopes[0][1]:.2f}，未见 1→2 过渡（低端不在扫描范围内）",
            "中",
            detail,
        )
    cross = next((mid for mid, s in slopes if s >= 1.5), None)
    if cross is None:
        return Verdict("E5", "regime 切换点 L*", "整个扫描范围内未见 1→2 过渡", "中", detail)
    return Verdict(
        "E5",
        "regime 切换点 L*",
        f"过渡发生在 L≈{cross:.0f}（取局部斜率首次 ≥1.5 处）",
        "中",
        detail,
    )


def infer_efficiency(l_star: float, p_flops: float, bw_bytes: float) -> dict[str, float]:
    """由实测 L* 反解有效效率。`η` 是唯一的校准量。"""
    if l_star <= 0 or p_flops <= 0 or bw_bytes <= 0:
        raise FrameworkError("L*、算力、带宽都要为正")
    ideal = 2.0 * p_flops / bw_bytes
    return {
        "ideal_l_star": ideal,
        "measured_l_star": l_star,
        "ratio": l_star / ideal,
        "p_eff": p_flops / (l_star / ideal),
        "bw_eff": bw_bytes / (l_star / ideal),
    }


# ---------------------------------------------------------------- 汇总


def verdicts_to_model(verdicts: list[Verdict]) -> dict[str, str]:
    """把各实验结论翻成对模型的直接指令。"""
    lookup = {v.experiment: v for v in verdicts}
    out: dict[str, str] = {}
    if "E1a" in lookup:
        answer = lookup["E1a"].answer
        out["b"] = (
            "b = c_dispatch · n_kernel，删掉 intercept 拟合列"
            if "是" in answer
            else "b 含非下发成分，按「计算段/固定块」分两类"
        )
    if "E2" in lookup:
        answer = lookup["E2"].answer
        if "两个 GEMM 等权" in answer:
            out["l2_width"] = "保留 value_dim，宽度 = d_head + d_v"
        elif "主项不是 GEMM" in answer:
            out["l2_width"] = "推翻 GEMM 主项，改查 softmax 与调度"
        else:
            out["l2_width"] = "待定"
    if "E3" in lookup:
        answer = lookup["E3"].answer
        if "台阶成立" in answer:
            out["ceil"] = "保留 ceil，BM/BN 可辨识"
        elif "不可测" in answer:
            out["ceil"] = "待定：先做 P1 降低同点噪声，再重跑 E3（不可测 ≠ 不存在）"
        else:
            out["ceil"] = "删掉 ceil，改回 L²/N"
    if "E4" in lookup:
        answer = lookup["E4"].answer
        if "线性" in answer:
            out["parallel"] = "N_work = B·H·ceil(L/(N·BM))·ceil(L/BN)"
        elif "平台" in answer:
            out["parallel"] = "N_work 外套 max(1, ceil(·/C))，并单独记平台宽度"
        else:
            out["parallel"] = "外循环并行，台阶间距给 C"
    if "E5" in lookup and "过渡发生在" in lookup["E5"].answer:
        out["regime"] = "由 L* 反解 P_eff 与 BW_eff，c 不再拟合"
    return out
