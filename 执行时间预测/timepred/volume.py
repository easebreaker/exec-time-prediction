"""集体通信体积，单位是字节，按每卡计。

v 的单位是秒/字节，把未被掩盖的有效带宽合在里面。换机器后 v 要重估。
体积若整体差一个常数倍，会被 v 吸收；随长度、batch、并行度变化的那一部分不会。

- CP：注意力输出 All-Gather。Query 按位置切开，KV 不切，每层一次。
  ((N - 1) / N) · n_layers · B · L · d · elem
- EmbP：查出的嵌入做 All-to-All。
  ((N - 1) / N^2) · B · n_ids · d · elem · n_layers
- TP：稠密矩阵的激活同步。注意力和 FFN 都切时每层两次，只切其中一段时每层一次。
  环状 AllReduce 为 2 · ((N - 1) / N) · 激活字节。
- EP：dispatch 与 combine 各一次 All-to-All。
  2 · ((N - 1) / N^2) · B · L · d · elem · n_layers
- PP：相邻 stage 的点对点激活。(S - 1) · B · L · d · elem
"""

from __future__ import annotations

from .schema import BlockSpec, FrameworkError, Load, Split


def _tokens(block: BlockSpec, load: Load) -> float:
    if block.kind in ("embedding", "cross"):
        return float(load.n_ids[block.name])
    return float(load.lengths[block.name])


def _activation_bytes(block: BlockSpec, load: Load, elem_bytes: int) -> float:
    return block.n_layers * load.batch * _tokens(block, load) * block.d_model * elem_bytes


def collective_volume_bytes(
    kind: str,
    blocks: list[BlockSpec],
    degree: int,
    load: Load,
    elem_bytes: int,
) -> float:
    if degree < 2 or not blocks:
        return 0.0
    per_rank = (degree - 1) / float(degree)
    if kind == "cp":
        total = 0.0
        for block in blocks:
            if block.kind != "attention":
                raise FrameworkError("CP 只切注意力")
            total += per_rank * _activation_bytes(block, load, elem_bytes)
        return total
    if kind == "embp":
        total = 0.0
        for block in blocks:
            if block.kind != "embedding":
                raise FrameworkError("EmbP 只切 Embedding 表")
            total += per_rank / float(degree) * _activation_bytes(block, load, elem_bytes)
        return total
    if kind == "ep":
        total = 0.0
        for block in blocks:
            if block.kind != "expert":
                raise FrameworkError("EP 只切 Expert")
            total += 2.0 * per_rank / float(degree) * _activation_bytes(block, load, elem_bytes)
        return total
    if kind == "tp":
        kinds = {block.kind for block in blocks}
        syncs = 2.0 if {"attention", "ffn"} <= kinds else 1.0
        reference = max(blocks, key=lambda block: _activation_bytes(block, load, elem_bytes))
        return syncs * 2.0 * per_rank * _activation_bytes(reference, load, elem_bytes)
    if kind == "pp":
        reference = max(blocks, key=lambda block: _tokens(block, load) * block.d_model)
        tensor = load.batch * _tokens(reference, load) * reference.d_model * elem_bytes
        return (degree - 1) * tensor
    raise FrameworkError(f"{kind} 没有通信体积")


def split_degree(splits: list[Split]) -> int:
    degrees = {split.degree for split in splits}
    if len(degrees) != 1:
        raise FrameworkError("同一种切分在同一次前向里使用同一个并行度")
    return degrees.pop()
