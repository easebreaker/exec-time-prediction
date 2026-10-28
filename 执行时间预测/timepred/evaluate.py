"""测试集只比较预测时间与实测时间，不更新参数。"""

from __future__ import annotations

from dataclasses import dataclass

from .predict import TimeModel, mean_abs_log_error
from .schema import FrameworkError, Sample, plan_token


@dataclass(frozen=True)
class EvalReport:
    n: int
    mean_abs_log_error: float
    mape: float
    n_loads_ranked: int
    rank_accuracy: float | None


def evaluate(model: TimeModel, samples: list[Sample]) -> EvalReport:
    if not samples:
        raise FrameworkError("测试样本为空")
    log_pairs: list[tuple[float, float]] = []
    ape: list[float] = []
    by_load: dict[tuple, list[tuple[Sample, float]]] = {}
    for sample in samples:
        predicted = model.predict(sample.load, sample.plan).time_s
        log_pairs.append((predicted, sample.time_s))
        ape.append(abs(predicted - sample.time_s) / sample.time_s)
        by_load.setdefault(sample.load.key(), []).append((sample, predicted))

    ranked = 0
    hits = 0
    for group in by_load.values():
        measured: dict[str, list[float]] = {}
        predicted_by_plan: dict[str, float] = {}
        for sample, predicted in group:
            token = plan_token(sample.plan)
            measured.setdefault(token, []).append(sample.time_s)
            predicted_by_plan[token] = predicted
        if len(measured) < 2:
            continue
        ranked += 1
        best_measured = min(
            measured,
            key=lambda token: (sum(measured[token]) / len(measured[token]), token),
        )
        best_predicted = min(predicted_by_plan, key=lambda token: (predicted_by_plan[token], token))
        hits += int(best_measured == best_predicted)
    accuracy = None if ranked == 0 else hits / ranked
    return EvalReport(
        n=len(samples),
        mean_abs_log_error=mean_abs_log_error(log_pairs),
        mape=sum(ape) / len(ape),
        n_loads_ranked=ranked,
        rank_accuracy=accuracy,
    )
