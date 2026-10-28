"""各块不切分时的基准时间 T_i^0。系数来自对照运行的分段计时。"""

from __future__ import annotations

from dataclasses import dataclass

from .linalg import nnls
from .schema import BlockSpec, FrameworkError, Load, ModelSpec


@dataclass(frozen=True)
class BlockBaseline:
    """与 features() 的列一一对应，系数非负。"""

    coefficients: tuple[float, ...]

    def time_s(self, row: list[float]) -> float:
        if len(row) != len(self.coefficients):
            raise FrameworkError("基准时间的特征维数和系数个数不一致")
        return sum(coef * value for coef, value in zip(self.coefficients, row))


def features(block: BlockSpec, load: Load) -> list[float]:
    """复杂度按块写死，只估系数。

    查表、交叉：(B · n_id, 1)
    注意力：(B · L^2 · d, B · L · d, 1)。平方项是 score，线性项是 KV。
    FFN：(B · L · d, 1)
    Expert：在 FFN 的形式上乘该步实际执行的 expert 份额。
    """
    batch = float(load.batch)
    width = float(block.d_model)
    if block.kind in ("embedding", "cross"):
        if block.name not in load.n_ids:
            raise FrameworkError(f"{block.name} 缺少 id 数")
        return [batch * float(load.n_ids[block.name]), 1.0]
    if block.name not in load.lengths:
        raise FrameworkError(f"{block.name} 缺少序列长度")
    length = float(load.lengths[block.name])
    if block.kind == "attention":
        return [batch * length * length * width, batch * length * width, 1.0]
    if block.kind == "ffn":
        return [batch * length * width, 1.0]
    if block.kind == "expert":
        share = float(load.expert_frac)
        return [share * batch * length * width, share]
    raise FrameworkError(f"未知的计算块类型 {block.kind}")


def baseline_time(block: BlockSpec, baseline: BlockBaseline, load: Load) -> float:
    return baseline.time_s(features(block, load))


def fit_baselines(
    spec: ModelSpec,
    rows_by_block: dict[str, list[list[float]]],
    times_by_block: dict[str, list[float]],
) -> dict[str, BlockBaseline]:
    fitted: dict[str, BlockBaseline] = {}
    for block in spec.blocks:
        rows = rows_by_block[block.name]
        times = times_by_block[block.name]
        width = len(rows[0])
        if len(rows) < width:
            raise FrameworkError(
                f"{block.name} 的对照负载组合有 {len(rows)} 组，"
                f"少于系数个数 {width}"
            )
        fitted[block.name] = BlockBaseline(tuple(nnls(rows, times)))
    return fitted
