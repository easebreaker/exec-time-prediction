"""把整套流程从头走一遍，每一步都把中间量打出来。

用法：`python3 -m timepred.flow_demo`

七步：
    0. 结构 → 阶段，附登记位
    1. 测量计划（88 条的一部分）
    2. 八个算子在合成数据上「该认出什么就认出什么」
    3. 真实数据：MTGR attn 的 13 点，幂律 / tile 计数 / 网络级 b 三种口径
    4. 切分拟合：κ、τ、v 可辨识，单长度退化
    5. 通信跨度：机内与跨机两档 v
    6. 网络预测：HSTU 分解版（融合开 / 关）与网络级截距

合成数据一律标注 `[合成]`；只有第 3、4 步用的是实测。
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from .experiment import (
    NetworkFacts,
    StageObservation,
    complexity_grid,
    fit_complexity,
    fit_split,
    kernel_count,
    predict_network,
    predict_stage,
    stage_time,
)
from .recipes import (
    hstu_stages,
    hstu_stages_tiled,
    hstu_stages_with_elemwise,
    moe_ffn_stages,
    retrieval_stages,
)
from .stage import FEATURE_ORDER, StageLoad, StageSpec, StageSplit, default_load, features, primary_term, volume_bytes

DATA = Path(__file__).resolve().parent.parent / "data" / "attn_mtgr_cp13.csv"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def observe(spec: StageSpec, coefficients: dict[str, float], loads, split: StageSplit, **kwargs):
    rows = []
    for load in loads:
        total, _, comm = stage_time(spec, coefficients, load, split, **kwargs)
        if split.kind == "none":
            rows.append(StageObservation(load, split, total))
        else:
            rows.append(StageObservation(load, split, total, volume_bytes(spec, load, split), comm))
    return rows


def synthetic_fit(spec: StageSpec, coefficients: dict[str, float], note: str) -> None:
    rows = observe(spec, coefficients, complexity_grid(spec), StageSplit("none"))
    report = fit_complexity(spec, rows)
    terms = "、".join(f"{name}={value:.3g}" for name, value in report.terms) or "无"
    print(f"  {spec.name:12s} {note}")
    print(f"    真值 : {coefficients}")
    print(f"    认出 : {terms}   MAPE={report.mape * 100:.4f}%   主项={primary_term(spec)}   匹配={report.matches_operator}   随 L={report.depends_on_length}")
    if report.reason:
        print(f"    说明 : {report.reason}")


def load_measured() -> dict[int, dict[int, float]]:
    """读实测的 13 点，返回 {并行度: {长度: 秒}}。"""
    table: dict[int, dict[int, float]] = {2: {}, 4: {}, 8: {}}
    with open(DATA, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            length = int(row["length"])
            for degree in (2, 4, 8):
                table[degree][length] = float(row[f"cp{degree}"]) / 1000.0
    return table


def measured_observations(
    degree: int,
    table: dict[int, dict[int, float]],
    *,
    span: str = "intra",
) -> tuple[StageSpec, list[StageObservation]]:
    spec = StageSpec("attn", "self_attn")
    split = StageSplit("cp_q", degree, span=span)
    rows = []
    for length, seconds in sorted(table[degree].items()):
        # heads 与 d_head 取 1，使 self_attn 的特征恰好等于 L²，
        # 与步骤 3、4 里 [1, L, L²] 的基同一口径。
        load = StageLoad(batch=1, length=length, n_head=1, d_head=1)
        rows.append(
            StageObservation(
                load,
                split,
                seconds,
                volume_bytes(spec, load, split),
                None,
            )
        )
    return spec, rows


def step0() -> None:
    rule("0 结构 → 阶段（登记位不可拟合，全是实现事实）")
    for name, builder in (
        ("HSTU（原配方）", hstu_stages),
        ("HSTU（分解出归一化）", hstu_stages_with_elemwise),
        ("HSTU（tile 计数）", hstu_stages_tiled),
        ("MoE 一层", moe_ffn_stages),
        ("召回一步", retrieval_stages),
    ):
        stages = builder()
        print(f"\n{name}：{len(stages)} 段，{kernel_count(stages)} 个 kernel")
        for spec in stages:
            flags = []
            if spec.fused:
                flags.append("fused")
            if spec.model != "power":
                flags.append(f"model={spec.model}({spec.block_m}x{spec.block_n})")
            if spec.sparse < 1.0:
                flags.append(f"σ={spec.sparse}")
            if spec.renorm:
                flags.append("renorm=1")
            print(
                f"    {spec.name:12s} {spec.operator:12s} 主项={primary_term(spec):14s} "
                f"kernel/层={spec.kernels_per_layer} {' '.join(flags)}"
            )


def step1() -> None:
    rule("1 测量计划（88 条里取 6 行；n_id 已是每样本次数）")
    from .plan import all_requests

    rows = list(all_requests())
    print(f"    合计 {len(rows)} 条")
    picked = rows[:3] + [r for r in rows if r.batch > 1][:2] + [r for r in rows if r.degree > 1][:1]
    for request in picked:
        print(
            f"    {request.id:14s} {request.model:8s} B={request.batch} L={request.length:5d} "
            f"n_real={request.n_real:6d} n_id={request.n_id:5d} split={request.split} x{request.degree}"
        )


def step2() -> None:
    rule("2 八个算子在合成数据上的识别 [合成]")
    synthetic_fit(StageSpec("embed", "lookup"), {"lookup": 1e-5, "intercept": 0.01}, "查表：随 n_id")
    synthetic_fit(StageSpec("proj", "linear"), {"linear": 1e-8, "intercept": 0.002}, "投影：随 token 数")
    synthetic_fit(StageSpec("attn", "self_attn"), {"self_attn": 2e-9, "intercept": 0.02}, "注意力：L²")
    synthetic_fit(
        StageSpec("attn", "self_attn", sdpa=True),
        {"attn_linear": 2e-6, "intercept": 0.02},
        "注意力 SDPA：线性项",
    )
    synthetic_fit(
        StageSpec("mix", "token_permute"),
        {"permute": 1e-8, "intercept": 0.001},
        "转置：按字节",
    )
    synthetic_fit(
        StageSpec("norm", "elemwise"),
        {"elemwise_linear": 4e-9, "elemwise_square": 2e-9, "intercept": 0.001},
        "归一化 + 对 L² 做逐元素（HSTU 的 SiLU on S）",
    )
    synthetic_fit(
        StageSpec("gate", "moe_gate"),
        {"gate_score": 3e-9, "gate_sort": 2e-6, "intercept": 0.004},
        "门控：随专家数",
    )
    synthetic_fit(
        StageSpec("score", "score_gemm"),
        {"score": 6e-11, "intercept": 0.003},
        "全库打分：随商品数",
    )
    fused = StageSpec("norm", "elemwise", fused=True)
    print(f"  fused=True 的段：全部列为 0 → {all(v == 0 for v in features(fused, default_load()).values())}")
    net = StageSpec("attn", "self_attn", intercept_mode="network")
    print(f"  intercept_mode=network 的段：intercept 列 = {features(net, default_load())['intercept']}")


def _fit_columns(rows: list[list[float]], times: list[float]) -> tuple[list[float], float]:
    """相对误差加权的正规方程（与 `实验数据-分析.md` 第六节同一目标）。

    `rows` 是 n × p 的设计矩阵，权重 1/y。这里的基只有 2–3 列且量级跨度大，
    用正规方程比迭代法稳定。
    """
    n = len(times)
    p = len(rows[0])
    weights = [1.0 / value for value in times]
    matrix = [
        [sum(weights[r] ** 2 * rows[r][i] * rows[r][j] for r in range(n)) for j in range(p)]
        + [sum(weights[r] ** 2 * rows[r][i] * times[r] for r in range(n))]
        for i in range(p)
    ]
    for col in range(p):
        pivot = max(range(col, p), key=lambda row: abs(matrix[row][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        if abs(matrix[col][col]) < 1e-300:
            raise ValueError("列退化")
        for row in range(p):
            if row != col:
                factor = matrix[row][col] / matrix[col][col]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col])]
    theta = [matrix[i][p] / matrix[i][i] for i in range(p)]
    preds = [sum(rows[r][i] * theta[i] for i in range(p)) for r in range(n)]
    mape = sum(abs(pred - true) / true for pred, true in zip(preds, times)) / n
    return theta, mape


def step3() -> None:
    rule("3 真实数据：MTGR attn 的 13 点（实测），三种口径对比")
    table = load_measured()
    print("  这是同一并行度下的阶段时间，不是未切分运行，所以不能用 fit_complexity；")
    print("  下面直接在 [1, L, L²] 与 [1, ceil(L/N/BM)·ceil(L/BM)] 两组基上做同样的相对误差加权。")
    print()
    print("  口径 A：b + θ₁L + θ₂L²（现状，3 参）")
    print("  口径 B：b + c·块数（2 参）")
    print()
    print(f"  {'并行度':6s} {'A: 3 参':>9s} {'B: 2 参':>9s}    A 的 b     B 的 b")
    block = 128
    for degree in (2, 4, 8):
        lengths = sorted(table[degree])
        times = [table[degree][length] for length in lengths]
        squares = [[1.0, float(length), float(length) ** 2] for length in lengths]
        tiles = [
            [1.0, float(math.ceil(length / degree / block) * math.ceil(length / block))]
            for length in lengths
        ]
        theta_a, mape_a = _fit_columns(squares, times)
        theta_b, mape_b = _fit_columns(tiles, times)
        print(
            f"  cp{degree:<4d} {mape_a * 100:8.2f}% {mape_b * 100:8.2f}%   "
            f"{theta_a[0] * 1000:8.2f}ms {theta_b[0] * 1000:8.2f}ms"
        )
    print()
    print("  读法一：2 个参数与 3 个参数打平，两者估出的 b 几乎相同（8.80/8.76、10.71/10.48、12.32/11.64）。")
    print("  读法二：块大小不可从数据辨识——无 ceil 时 块数 ∝ L²/N，模型退化为幂律；改 BM 只把单价按 BM² 反比缩放。")
    print("  读法三：tile 模型的价值在解释机制，不在提高精度；精度要靠把 b 直接测出来减掉。")
    print()
    print("  把 b 当作已测出、从每个点减掉后的局部指数（cp2，b = 8.80 ms）：")
    floor = 8.80 / 1000.0
    lengths = sorted(table[2])
    for left, right in zip(lengths, lengths[1:]):
        if left < 2048:
            continue
        t_left = table[2][left] - floor
        t_right = table[2][right] - floor
        if t_left <= 0 or t_right <= 0:
            continue
        exponent = math.log(t_right / t_left) / math.log(right / left)
        print(f"    {left:>6d} → {right:<6d} 局部指数 {exponent:5.2f}（幂律整体给 2.00）")


def step4() -> None:
    rule("4 切分拟合：κ、τ、v 能否分开（实测，含单长度退化）")
    table = load_measured()
    spec = StageSpec("attn", "self_attn")
    # 真实流程里复杂度来自「未切分」系列。这批数据没有 cp1，
    # 于是以 cp2 上 A 口径的三项式当复杂度基线，并明确标注这一点。
    lengths = sorted(table[2])
    times = [table[2][length] for length in lengths]
    theta, mape = _fit_columns(
        [[1.0, float(length), float(length) ** 2] for length in lengths], times
    )
    from .experiment import ComplexityReport

    complexity = ComplexityReport(
        stage="attn",
        operator="self_attn",
        terms=(("self_attn", theta[2]), ("intercept", theta[0])),
        mape=mape,
        identified=True,
        matches_operator=True,
        depends_on_length=True,
        length_checked=True,
        width_log_slope=None,
        reason="",
    )
    print(f"  复杂度基线（借用 cp2 的 A 口径，非未切分实测）：b={theta[0] * 1000:.2f} ms，θ₂={theta[2]:.3e}")

    degrees = []
    for degree in (2, 4, 8):
        _, group = measured_observations(degree, table)
        degrees.extend(group)
    fitted = fit_split(spec, complexity, degrees)
    print(
        f"  三档 × 13 长度：κ={fitted.kappa:.3g}  τ={fitted.tau_s * 1000:.3f} ms  "
        f"v={fitted.v_s_per_byte:.3g} s/B  分开={fitted.identified}  MAPE={fitted.train_mape * 100:.2f}%"
    )
    print(f"    体积公式与实测字节是否一致：{fitted.volume_matches}（这批没有记通信字节，故为空）")
    one = [obs for obs in degrees if obs.load.length == 16384]
    collapsed = fit_split(spec, complexity, one)
    print(f"  只在一条长度上改并行度：分开={collapsed.identified}  说明={collapsed.reason or '无'}")
    print(f"    κ={collapsed.kappa:.3g}  τ={collapsed.tau_s * 1000:.3f} ms  v={collapsed.v_s_per_byte:.3g}")


def step5() -> None:
    rule("5 通信跨度：机内与跨机两档 v 可以分开 [合成]")
    spec = StageSpec("comm", "moe_comm")
    truth = {"moe_route": 2e-6, "intercept": 0.0}
    rows = []
    for span, v in (("intra", 6e-9), ("cross", 5e-8)):
        for degree in (2, 4, 8):
            split = StageSplit("ep", degree, span=span)
            for length in (256, 1024):
                load = StageLoad(batch=2, length=length, n_head=8, d_head=64)
                total, _, comm = stage_time(
                    spec, truth, load, split, tau_s=5e-5, v_s_per_byte=v
                )
                rows.append(StageObservation(load, split, total, volume_bytes(spec, load, split), comm))
    complexity = fit_complexity(spec, observe(spec, truth, complexity_grid(spec), StageSplit("none")))
    fit = fit_split(spec, complexity, rows)
    print(f"  真值：v(机内)=6e-9  v(跨机)=5e-8  τ=5e-5")
    print(f"  认出：v(机内)={fit.v_s_per_byte:.3g}  v(跨机)={fit.v_cross_s_per_byte:.3g}  τ={fit.tau_s:.3g}")
    print(f"  倍数：{(fit.v_cross_s_per_byte or 0) / max(fit.v_s_per_byte, 1e-30):.2f}x（手册估约 10x）")
    print(f"  分开={fit.identified}  MAPE={fit.train_mape * 100:.3f}%")
    one_span = [obs for obs in rows if obs.split.span == "intra"]
    collapsed = fit_split(spec, complexity, one_span)
    print(f"  只给机内样本：跨机那一档={collapsed.v_cross_s_per_byte}")


def step6() -> None:
    rule("6 网络预测：融合开 / 关，以及网络级截距 [合成]")
    base = StageLoad(batch=1, length=2048, n_head=8, d_head=64, hidden=512, width=128)
    coefficients = {
        "lookup": 1e-6,
        "linear": 3e-10,
        "self_attn": 6e-11,
        "elemwise_linear": 4e-9,
        "elemwise_square": 2e-10,
    }
    for label, stages in (
        ("原配方（3 段）", hstu_stages()),
        ("分解版（融合开启）", hstu_stages_with_elemwise()),
        ("分解版（融合关闭）", tuple(
            StageSpec(s.name, s.operator, s.n_layers, fused=False) for s in hstu_stages_with_elemwise()
        )),
    ):
        total = 0.0
        for spec in stages:
            value, _, _ = stage_time(spec, coefficients, base, StageSplit("none"))
            total += value
        print(f"  {label:22s} 合计 {total * 1000:8.3f} ms   kernel 数 {kernel_count(stages)}")

    # 网络级截距：段内 intercept 归零，b 由 kernel 数算出
    stages = tuple(
        StageSpec(s.name, s.operator, s.n_layers, intercept_mode="network")
        for s in hstu_stages()
    )
    complexities = {}
    for spec in stages:
        rows = observe(spec, coefficients, complexity_grid(spec), StageSplit("none"))
        complexities[spec.name] = fit_complexity(spec, rows)
    facts = NetworkFacts(fixed_s=1e-4, dispatch_s=8e-6, sync_s=2e-5)
    budget = predict_network(stages, complexities, {}, base, network=facts)
    print(f"\n  网络级 b：{facts.dispatch_s * 1e6:.0f} μs/kernel × {budget.n_kernel} kernel "
          f"+ {facts.fixed_s * 1e3:.2f} ms + 同步 = {budget.network_s * 1000:.3f} ms")
    print(f"  逐段之和：{budget.time_s * 1000:.3f} ms（段内 intercept 已归零）")


def main() -> None:
    print("全过程演示。标 [合成] 的步骤用公式生成的时间，只核对流程；第 3、4 步用实测。")
    step0()
    step1()
    step2()
    step3()
    step4()
    step5()
    step6()
    step7()
    rule("完")


def step7() -> None:
    rule("7 官方 HSTU 单层的形状核对（需要 torch；没有则跳过）")
    probe = Path(__file__).resolve().parent.parent.parent / "_vendor" / "hstu_probe.py"
    if not probe.exists():
        print(f"  找不到 {probe}，跳过。")
        return
    import os
    import subprocess

    # 不在本进程里 import torch：macOS 沙箱下 libomp 会因共享内存失败而 abort，
    # 放到子进程里跑，失败只影响这一步。
    env = dict(os.environ, OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print("  跳过：核对脚本没跑通。原因二选一，按提示处理：")
        for line in (result.stdout or "").splitlines():
            if line.startswith(("没有", "  git clone", "      _vendor")):
                print("   ", line.strip())
        return
    for line in result.stdout.splitlines():
        if line.startswith(("结论", "  ·", "  linear", "  self_attn", "  qk", "  elemwise", "  p·v", "  官方")):
            print(line)


if __name__ == "__main__":
    main()
