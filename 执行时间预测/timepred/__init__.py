"""给定一个模型上的多种切分和多组负载，预测一次前向的执行时间。

结构先固定。各块基准时间来自对照运行的分段计时。
每种切分用自己的样本估计计算加速 g(N) = N / (1 + κ(N - 1))，
以及未掩盖通信 τ + v·Vol。混合方案把各块时间相加。
显存不超过单卡容量的方案里，取预测时间最短的。
"""

from __future__ import annotations

from .evaluate import EvalReport, evaluate
from .fit import FitReport, fit
from .formula import compute_speedup, slide_time, speedup_upper_bound
from .holdout import Holdout, holdout_by_load
from .io import load_samples, parse_plan, write_template
from .predict import Breakdown, SchemeParams, TimeModel
from .schema import (
    BlockSpec,
    FrameworkError,
    Load,
    ModelSpec,
    Plan,
    Sample,
    Split,
)

__all__ = [
    "BlockSpec",
    "Breakdown",
    "EvalReport",
    "FitReport",
    "FrameworkError",
    "Holdout",
    "Load",
    "ModelSpec",
    "Plan",
    "Sample",
    "SchemeParams",
    "Split",
    "TimeModel",
    "compute_speedup",
    "evaluate",
    "fit",
    "holdout_by_load",
    "load_samples",
    "parse_plan",
    "slide_time",
    "speedup_upper_bound",
    "write_template",
]
