"""预测式。

不切分的块留在基准时间里。被一种切分切到的块写成

    T = T_cut / g + τ + v · Vol

g 由并行度 N 和该切分的 κ 决定：

    g(N) = N / (1 + κ (N - 1))

κ = 0 时 g = N。κ = 1 时 g = 1。κ 再大，计算本身会慢于不切分。
因此 g 不超过 N。通信体积随负载变化，所以同一切分的有效耗时仍随负载变化。

只切一组块、并且并行度相同的时候，上式就是修订稿里的

    T(p, x) = T_0 [(1 - r) + r / g + c]

其中 r 是被切块占基准时间的比例，c = (τ + v·Vol) / T_0。
多种切分同时存在时，各块时间相加，不再合成一个 g。
流水并行不使用 g，计算项乘 bubble 对应的时间系数 (S + M - 1) / (M S)。
"""

from __future__ import annotations

import math


def compute_speedup(kappa: float, degree: int) -> float:
    if degree < 1:
        raise ValueError("并行度至少为 1")
    if kappa < 0.0:
        raise ValueError("κ 不能为负")
    if degree == 1:
        return 1.0
    return degree / (1.0 + kappa * (degree - 1))


def pipeline_factor(n_stages: int, n_microbatches: int) -> float:
    """流水时间 / 该段不切分时间。(S + M - 1) / (M S)。"""
    if n_stages < 1 or n_microbatches < 1:
        raise ValueError("stage 与 micro-batch 至少为 1")
    return (n_stages + n_microbatches - 1) / (n_microbatches * n_stages)


def linear_cut(t0_cut_s: float, kind: str, degree: int, n_microbatches: int) -> tuple[float, float]:
    """把被切块的计算时间写成 offset + κ · feature。

    与 T_cut / g(κ) 是同一件事，拟合时用这个线性形式。
    流水并行没有 κ，feature 为 0，offset 已经乘上流水系数。
    """
    if kind == "pp":
        return t0_cut_s * pipeline_factor(degree, n_microbatches), 0.0
    return (
        t0_cut_s / degree,
        t0_cut_s * (degree - 1) / degree,
    )


def slide_time(t0_s: float, r: float, g: float, c: float) -> float:
    """T_0 [(1 - r) + r / g + c]。只用于单一切分的核对。"""
    if t0_s < 0.0 or g <= 0.0:
        raise ValueError("基准时间和 g 要为正")
    return t0_s * ((1.0 - r) + r / g + c)


def speedup_upper_bound(serial_fraction: float) -> float:
    """未切分时间占比 s 给出的加速比上界 1 / s。整网都被切到时没有这个上界。"""
    if serial_fraction < 0.0:
        raise ValueError("未切分占比不能为负")
    if serial_fraction == 0.0:
        return math.inf
    return 1.0 / serial_fraction
