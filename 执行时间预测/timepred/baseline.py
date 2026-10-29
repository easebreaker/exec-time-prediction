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
    """各块的计算量。层数写在块上，不随样本变化，收进系数，不出现在特征里。

    符号只表示该块真正读到的量。用户序列长度记为 L，只进入注意力。
    token mixer 的平方维是组内 token 数 T，不读 L。

    查表、交叉：B · n_id
        B 是 batch。n_id 是该块查询的 id 数。与 L、T 无关。
    注意力：B · L² · d ，以及 B · L · d
        L 是该注意力看到的序列长度。d 是 d_model。
        第一项是 score，第二项是 KV。
    token mixer：B · R · T² · d
        T 是一组里被混合的 token 数。R 是这样的组数。
        d 是 d_model。长序列长度 L 不进入这一项。
    FFN：B · N_tok · d
        N_tok 是该 FFN 看到的 token 数。隐层宽度收进系数。
        接在 mixer 后面时，N_tok 取 T，不取 L。
    Expert：s · B · N_tok · d
        s 是该步实际执行的 expert 份额。其余与 FFN 相同。
    """
    batch = float(load.batch)
    width = float(block.d_model)
    if block.kind in ("embedding", "cross"):
        if block.name not in load.n_ids:
            raise FrameworkError(f"{block.name} 缺少 id 数")
        return [batch * float(load.n_ids[block.name]), 1.0]
    if block.kind == "token_mixer":
        if block.name not in load.n_groups:
            raise FrameworkError(f"{block.name} 缺少组数 R")
        if block.name not in load.lengths:
            raise FrameworkError(f"{block.name} 缺少组内 token 数 T")
        groups = float(load.n_groups[block.name])
        tokens = float(load.lengths[block.name])
        return [batch * groups * tokens * tokens * width, 1.0]
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
