"""原理性测定方案：把测量表打出来，并在两组「假想结果」上跑判定树。

用法：
    python3 -m timepred.principle_demo

第 1 段打实验清单（真实要跑的东西）。
第 2 段用合成读数演示判定树：同一个判据，输入不同形状的数据，给出不同结论。
第 3 段把现有 13 点实测喂进去，看它诚实地报「不够」。
"""

from __future__ import annotations

import math

from .flow_demo import load_measured
from .principle import (
    Reading,
    by_experiment,
    experiment_plan,
    identify_e1a,
    identify_e2,
    identify_e3,
    identify_e4,
    identify_e5,
    infer_efficiency,
    verdicts_to_model,
)

BM = 128


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def blocks(length: int, degree: int, block: int = BM) -> int:
    return math.ceil(length / degree / block) * math.ceil(length / block)


def show_plan() -> None:
    rule("1 测量清单（31 条；每条建议重复 5 次取中位数）")
    grouped = by_experiment(experiment_plan())
    for name, trials in grouped.items():
        print(f"\n{name}（{len(trials)} 条）")
        for trial in trials:
            split = "none" if trial.split.kind == "none" else f"{trial.split.kind}×{trial.split.degree}"
            print(
                f"  {trial.id:16s} L={trial.load.length:<6} B·H={trial.load.batch * trial.load.n_head:<3} "
                f"d={trial.load.d_head:<4} split={split:10s} {trial.note}"
            )


def show_decision_tree() -> None:
    rule("2 判定树：同一判据，输入不同形状的数据 → 不同结论 [合成]")

    print("\n【E1a】b 是否 ∝ kernel 数")
    linear = [(1, 2.2e-3), (2, 4.4e-3), (4, 8.8e-3)]
    print(f"  输入 b = {[round(b * 1e3, 2) for _, b in linear]} ms（严格倍比）")
    print("    " + identify_e1a(linear).answer)
    offset = [(1, 3.0e-3), (2, 4.4e-3), (4, 7.2e-3)]
    print(f"  输入 b = {[round(b * 1e3, 2) for _, b in offset]} ms（含常数项）")
    print("    " + identify_e1a(offset).answer)

    print("\n【E2】L² 项宽度")
    print("  输入 1 : 1.5 : 1.5（两 GEMM 等权）")
    print("    " + identify_e2([(64, 64, 1.0), (128, 64, 1.5), (64, 128, 1.5)]).answer)
    print("  输入 1 : 1 : 1（与头维无关）")
    print("    " + identify_e2([(64, 64, 1.0), (128, 64, 1.0), (64, 128, 1.0)]).answer)

    print("\n【E3】块数台阶（cp8，L=8192/8320/8448，块数 512/547/576）")
    lengths = [8192, 8320, 8448]
    steps = [blocks(L, 8) for L in lengths]
    print(f"  块数 = {steps}（相邻比 {steps[1] / steps[0]:.3f}、{steps[2] / steps[1]:.3f}）")
    for sigma, label in ((0.02, "同点噪声 2%（重复 10 次）"), (0.115, "同点噪声 11.5%（只测一次）")):
        stepped = [Reading(L, 0.3e-3 + 7.5e-6 * s, sigma * (0.3e-3 + 7.5e-6 * s)) for L, s in zip(lengths, steps)]
        smooth = [
            Reading(L, 0.3e-3 + 7.5e-6 * steps[0] * (L / lengths[0]) ** 2 * 8 / 8,
                    sigma * (0.3e-3 + 7.5e-6 * steps[0] * (L / lengths[0]) ** 2))
            for L in lengths
        ]
        print(f"\n  输入阶梯，{label}")
        print("    " + identify_e3(stepped, steps).answer)
        print(f"  输入平滑，{label}")
        print("    " + identify_e3(smooth, steps).answer)

    print("\n【E4】并行结构（B·H = 1…32）")
    xs = [1, 2, 4, 8, 16, 32]
    print("  输入线性：T ∝ B·H")
    print("    " + identify_e4([Reading(x, 0.8e-3 * x, 0.02e-3) for x in xs]).answer)
    print("  输入平台：B·H ≤ 8 时平坦，之后才涨")
    plateau = [Reading(x, 0.8e-3 * max(1.0, x / 8), 0.02e-3) for x in xs]
    print("    " + identify_e4(plateau).answer)

    print("\n【E5】regime 切换（L = 256…65536）")
    ladder = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    l_star = 1200.0
    mixed = [
        Reading(L, 0.5e-3 + 8e-3 * L / l_star if L < l_star else 0.5e-3 + 8e-3 * (L / l_star) ** 2, 0.05e-3)
        for L in ladder
    ]
    print(f"  输入：L* 设为 {l_star:.0f}")
    print("    " + identify_e5(mixed).answer)
    eff = infer_efficiency(l_star, 245e12, 1.6e12)
    print(
        f"  反解：理论 L* = {eff['ideal_l_star']:.0f}，实测 {eff['measured_l_star']:.0f}，"
        f"比值 {eff['ratio']:.1f}x → P_eff = {eff['p_eff']:.3g} FLOP/s，BW_eff = {eff['bw_eff']:.3g} B/s"
    )


def show_existing() -> None:
    rule("3 现有 13 点：判定树会说什么？")
    table = load_measured()
    lengths = sorted(table[2])
    print("\n现有数据只有 N≥2、且 L 全在 128 的块格上，因此：")
    print("  E1a 缺（没有层数变化）        → 无法判定 b 的物理来源")
    print("  E2 缺（没有改头维）            → 无法确认 L² 项宽度")
    print("  E3 缺（全是格点）              → 台阶不可辨识")
    print("  E4 缺（B·H 固定为 8）          → 无法定并行结构")
    print("  E5 缺 N=1                      → 无法分离切分效应")

    readings = [Reading(L, table[2][L], 1.05e-3) for L in lengths]
    verdict = identify_e5(readings)
    print(f"\n  唯一能勉强跑的是 E5（把 cp2 当成「未切分」的近似，抖动用小段 σ=1.05 ms）：")
    print(f"    {verdict.answer}")
    for line in verdict.detail:
        print("      " + line)

    print("\n  结论：现有数据能给出的最强陈述是——")
    print("    · b ≈ 8.8 ms 量级、且与 L 无关（13 个点跨 64 倍仍成立）")
    print("    · L=256 三档几乎相同（9.75/11.68/11.76），与「不足一波」一致")
    print("    · 局部斜率确实随 L 上升，但无法区分「b 稀释」与「regime 切换」")
    print("  这三条都不足以定位形式，所以需要 §1 的 31 条。")


def show_model() -> None:
    rule("4 判定汇总：结论如何直接改模型")
    ladder = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
    l_star = 1200.0
    e5 = identify_e5(
        [
            Reading(L, 0.5e-3 + 8e-3 * L / l_star if L < l_star else 0.5e-3 + 8e-3 * (L / l_star) ** 2, 0.05e-3)
            for L in ladder
        ]
    )
    e3_steps = [blocks(L, 8) for L in (8192, 8320, 8448)]
    e3_times = [0.3e-3 + 7.5e-6 * s for s in e3_steps]
    e3 = identify_e3(
        [Reading(8192 + 128 * i, t, 0.02 * t) for i, t in enumerate(e3_times)],
        e3_steps,
    )
    verdicts = [
        identify_e1a([(1, 2.2e-3), (2, 4.4e-3), (4, 8.8e-3)]),
        identify_e2([(64, 64, 1.0), (128, 64, 1.5), (64, 128, 1.5)]),
        e3,
        identify_e4([Reading(x, 0.8e-3 * x, 0.02e-3 * x) for x in (1, 2, 4, 8, 16, 32)]),
        e5,
    ]
    for verdict in verdicts:
        print(f"\n{verdict.line()}")
        for line in verdict.detail:
            print("      " + line)
    print("\n对模型的直接指令：")
    for key, value in verdicts_to_model(verdicts).items():
        print(f"  {key:10s} {value}")


def main() -> None:
    print("原理性测定方案演示。第 2 段用合成读数，第 3 段用现有实测。")
    show_plan()
    show_decision_tree()
    show_existing()
    show_model()
    rule("完")


if __name__ == "__main__":
    main()
