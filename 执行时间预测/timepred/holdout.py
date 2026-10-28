"""按负载组合留出测试集。同一 batch、同一组序列长度的各种切分留在同一侧。"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import Sample


@dataclass(frozen=True)
class Holdout:
    train: list[Sample]
    test: list[Sample]
    test_keys: tuple


def holdout_by_load(
    samples: list[Sample],
    test_keys: set | None = None,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Holdout:
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("测试组合的比例要在 0 到 1 之间")
    keys = sorted({sample.load.key() for sample in samples})
    if test_keys is None:
        order = list(keys)
        random.Random(seed).shuffle(order)
        if len(order) <= 1:
            chosen: set = set()
        else:
            n_test = int(round(len(order) * test_fraction))
            n_test = min(max(n_test, 1), len(order) - 1)
            chosen = set(order[:n_test])
    else:
        chosen = set(test_keys)
    train: list[Sample] = []
    test: list[Sample] = []
    for sample in samples:
        if sample.load.key() in chosen:
            test.append(sample)
        else:
            train.append(sample)
    return Holdout(train, test, tuple(sorted(chosen)))
