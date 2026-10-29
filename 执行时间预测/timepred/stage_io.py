"""阶段实验的测量表。每一行是一个阶段的一次运行。

标准列不够用的轴写成 x_<字段名>，例如 HyFormer 的 x_short_len。
通信字节数和通信时间可以留空。留空表示这次没有单独量到通信。
"""

from __future__ import annotations

import csv
from pathlib import Path

from .experiment import StageObservation
from .schema import FrameworkError
from .stage import StageLoad, StageSpec, StageSplit

_COLUMNS = (
    "batch",
    "length",
    "n_real",
    "n_id",
    "n_query",
    "length_kv",
    "tokens",
    "groups",
    "width",
    "hidden",
    "n_head",
    "d_head",
    "split",
    "degree",
    "micro",
    "time_s",
    "comm_bytes",
    "comm_time_s",
)


def headers(spec: StageSpec) -> list[str]:
    extra = sorted(field for field in spec.axes.values() if field not in _COLUMNS)
    return list(_COLUMNS) + [f"x_{field}" for field in extra]


def write_stage_template(path: str | Path, spec: StageSpec) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(headers(spec))


def load_stage_samples(path: str | Path) -> list[StageObservation]:
    observations: list[StageObservation] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "time_s" not in reader.fieldnames:
            raise FrameworkError("阶段测量表至少要有 time_s")
        for line_number, row in enumerate(reader, start=2):
            try:
                observations.append(_row(row))
            except FrameworkError as exc:
                raise FrameworkError(f"第 {line_number} 行: {exc}") from exc
    if not observations:
        raise FrameworkError("阶段测量表里没有数据行")
    return observations


def _cell(row: dict[str, str | None], name: str) -> str:
    value = row.get(name)
    return "" if value is None else value.strip()


def _required_int(text: str, label: str) -> int:
    if text == "":
        raise FrameworkError(f"缺少 {label}")
    try:
        return int(text)
    except ValueError as exc:
        raise FrameworkError(f"{label} 不是整数") from exc


def _optional_int(text: str, label: str) -> int | None:
    if text == "":
        return None
    return _required_int(text, label)


def _optional_float(text: str, label: str) -> float | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise FrameworkError(f"{label} 不是数字") from exc


def _row(row: dict[str, str | None]) -> StageObservation:
    extras = {}
    for name, value in row.items():
        if name is None or not name.startswith("x_"):
            continue
        parsed = _optional_int(_cell(row, name), name)
        if parsed is not None:
            extras[name[2:]] = parsed
    n_real = _optional_int(_cell(row, "n_real"), "n_real")
    load = StageLoad(
        batch=_required_int(_cell(row, "batch"), "batch"),
        length=_required_int(_cell(row, "length"), "length"),
        n_real=n_real,
        n_id=_required_int(_cell(row, "n_id"), "n_id"),
        n_query=_required_int(_cell(row, "n_query"), "n_query"),
        length_kv=_required_int(_cell(row, "length_kv"), "length_kv"),
        tokens=_required_int(_cell(row, "tokens"), "tokens"),
        groups=_required_int(_cell(row, "groups"), "groups"),
        width=_required_int(_cell(row, "width"), "width"),
        hidden=_required_int(_cell(row, "hidden"), "hidden"),
        n_head=_required_int(_cell(row, "n_head"), "n_head"),
        d_head=_required_int(_cell(row, "d_head"), "d_head"),
        extras=extras,
    )
    kind = _cell(row, "split") or "none"
    degree = _optional_int(_cell(row, "degree"), "degree") or 1
    micro = _optional_int(_cell(row, "micro"), "micro") or 1
    time_s = _optional_float(_cell(row, "time_s"), "time_s")
    if time_s is None:
        raise FrameworkError("缺少 time_s")
    return StageObservation(
        load=load,
        split=StageSplit(kind, degree, micro),
        time_s=time_s,
        comm_bytes=_optional_float(_cell(row, "comm_bytes"), "comm_bytes"),
        comm_time_s=_optional_float(_cell(row, "comm_time_s"), "comm_time_s"),
    )
