"""测量表。每一行是一次已经实现的前向。

列：batch, time_s, peak_mem_bytes, expert_frac, plan，
以及每个块的 len_<块名>、ids_<块名>、t_<块名>。

plan 的写法是块名=切分:并行度，多块用分号隔开。对照写 none。
流水并行写成 attention=pp:4:32，最后一段是 micro-batch 数。
峰值显存和分段时间可以留空。分段时间只在对照运行里用于拟合基准。
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import FrameworkError, Load, ModelSpec, Plan, Sample, Split, ensure_plan


def headers(spec: ModelSpec) -> list[str]:
    columns = ["batch", "time_s", "peak_mem_bytes", "expert_frac", "plan"]
    for block in spec.blocks:
        columns.append(f"len_{block.name}")
        columns.append(f"ids_{block.name}")
        columns.append(f"t_{block.name}")
    return columns


def write_template(path: str | Path, spec: ModelSpec) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(headers(spec))


def parse_plan(text: str) -> Plan:
    text = text.strip()
    if text in ("", "none", "对照"):
        return Plan({})
    splits: dict[str, Split] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise FrameworkError(f"无法解析方案 {part}，格式是 块名=切分:并行度")
        name, body = part.split("=", 1)
        bits = body.split(":")
        kind = bits[0].strip()
        try:
            degree = int(bits[1]) if len(bits) > 1 else 1
            micro = int(bits[2]) if len(bits) > 2 else 1
        except ValueError as exc:
            raise FrameworkError(f"无法解析方案 {part}") from exc
        splits[name.strip()] = Split(kind, degree, micro)
    return Plan(splits)


def load_samples(path: str | Path, spec: ModelSpec) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "batch" not in reader.fieldnames or "time_s" not in reader.fieldnames:
            raise FrameworkError("测量表至少要有 batch 和 time_s 两列")
        for line_number, row in enumerate(reader, start=2):
            samples.append(_row_to_sample(spec, row, line_number))
    if not samples:
        raise FrameworkError("测量表里没有数据行")
    return samples


def _cell(row: dict[str, str | None], name: str) -> str:
    value = row.get(name)
    return "" if value is None else value.strip()


def _optional_int(text: str, label: str) -> int | None:
    if text == "":
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise FrameworkError(f"{label} 不是整数: {text}") from exc


def _optional_float(text: str, label: str) -> float | None:
    if text == "":
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise FrameworkError(f"{label} 不是数字: {text}") from exc


def _row_to_sample(spec: ModelSpec, row: dict[str, str | None], line_number: int) -> Sample:
    try:
        batch = int(_cell(row, "batch"))
        time_s = float(_cell(row, "time_s"))
    except ValueError as exc:
        raise FrameworkError(f"第 {line_number} 行的 batch 或 time_s 无法解析") from exc
    expert_text = _cell(row, "expert_frac")
    try:
        expert_frac = 1.0 if expert_text == "" else float(expert_text)
    except ValueError as exc:
        raise FrameworkError(f"第 {line_number} 行的 expert_frac 无法解析") from exc
    lengths: dict[str, int] = {}
    n_ids: dict[str, int] = {}
    block_times: dict[str, float] = {}
    for block in spec.blocks:
        length = _optional_int(_cell(row, f"len_{block.name}"), f"len_{block.name}")
        if length is not None:
            lengths[block.name] = length
        ids = _optional_int(_cell(row, f"ids_{block.name}"), f"ids_{block.name}")
        if ids is not None:
            n_ids[block.name] = ids
        segment = _optional_float(_cell(row, f"t_{block.name}"), f"t_{block.name}")
        if segment is not None:
            block_times[block.name] = segment
    peak = _optional_int(_cell(row, "peak_mem_bytes"), "peak_mem_bytes")
    try:
        plan = parse_plan(_cell(row, "plan") or "none")
        ensure_plan(spec, plan)
        return Sample(
            load=Load(batch, lengths, n_ids, expert_frac),
            plan=plan,
            time_s=time_s,
            block_times_s=block_times or None,
            peak_mem_bytes=peak,
        )
    except FrameworkError as exc:
        raise FrameworkError(f"第 {line_number} 行: {exc}") from exc
