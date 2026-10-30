"""一个模型写成阶段序列。每个阶段只做一种算子。

整次前向的时间是各阶段时间之和。复杂度、切分加速比和通信按阶段估计，
不按 HyFormer、LONGER、HSTU、MTGR 的名字估计。这些模型的差别是阶段
怎么排列，以及长度轴接到哪一个输入。

算子和它在单卡上的主项：

- lookup 查表：B · n_id
- linear 线性投影：N_tok · d · d_hid。N_tok 由 token_axis 决定。
- self_attn 自注意力：B · n_head · L² · d_head。L 是该批的填充长度。
  SDPA 的实测更接近 L 的 1 到 1.3 次，所以另有一次项
  attn_linear = B · n_head · L · d_head。两项同时在时，有效次数落在 1 和 2 之间。
  切分仍只缩小平方项。只切 Query、KV 留整段时，一次项还要读整段 KV，不除以并行度。
- cross_attn 交叉注意力：B · N_q · L_kv · d
- group_attn 组内注意力：B · R · T² · d。groups_follow_length 时 R = L / T。
- token_gemm token 轴矩阵乘：与组内注意力同一单项。R 固定时不读 L。
- token_permute 转置混合：B · T · d，按字节计，没有 T²。

切分只缩小它切得到的那一项。自注意力只切 Query、KV 留整段时，平方项除以
并行度；Query 和 KV 都切开时，平方项除以并行度的平方。线性投影按隐藏维切
时除以并行度。按组切开且组之间没有归约时，没有集合通信。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .formula import pipeline_factor
from .schema import FrameworkError

OPERATORS = (
    "lookup",
    "linear",
    "self_attn",
    "cross_attn",
    "group_attn",
    "token_gemm",
    "token_permute",
)
TOKEN_AXES = ("real", "group", "fixed", "query")
SPLIT_PATTERNS = (
    "none",
    "cp_q",
    "cp_qk",
    "shard_kv",
    "shard_q",
    "tp",
    "embp",
    "shard_r",
    "shard_t",
    "pp",
)
FEATURE_ORDER = (
    "lookup",
    "linear",
    "self_attn",
    "attn_linear",
    "cross_attn",
    "token_square",
    "token_linear",
    "permute",
    "intercept",
)
PRIMARY_TERM = {
    "lookup": "lookup",
    "linear": "linear",
    "self_attn": "self_attn",
    "cross_attn": "cross_attn",
    "group_attn": "token_square",
    "token_gemm": "token_square",
    "token_permute": "permute",
}
LEGAL_SPLITS = {
    "lookup": ("none", "embp"),
    "linear": ("none", "tp", "pp"),
    "self_attn": ("none", "cp_q", "cp_qk"),
    "cross_attn": ("none", "shard_kv", "shard_q"),
    "group_attn": ("none", "shard_r"),
    "token_gemm": ("none", "shard_r", "shard_t"),
    "token_permute": ("none", "shard_t"),
}
# 正数表示这一项的单卡时间除以 N ** power，再乘 (1 + κ (N - 1))。
SCALING_POWER = {
    "cp_q": {"self_attn": 1},
    "cp_qk": {"self_attn": 2},
    "shard_kv": {"cross_attn": 1},
    "shard_q": {"cross_attn": 1},
    "tp": {"linear": 1, "token_linear": 1},
    "embp": {"lookup": 1},
    "shard_r": {"token_square": 1, "token_linear": 1},
    "shard_t": {"token_square": 1, "permute": 1, "token_linear": 1},
}
OPERATOR_LABEL = {
    "lookup": "查表",
    "linear": "线性投影",
    "self_attn": "自注意力",
    "cross_attn": "交叉注意力",
    "group_attn": "组内注意力",
    "token_gemm": "token 轴矩阵乘",
    "token_permute": "转置混合",
}
SPLIT_LABEL = {
    "none": "不切分",
    "cp_q": "CP（只切 Query）",
    "cp_qk": "CP（Query 和 KV 都切）",
    "shard_kv": "切 KV",
    "shard_q": "切 Query",
    "tp": "TP",
    "embp": "EmbP",
    "shard_r": "按组切开",
    "shard_t": "按组内 token 切开",
    "pp": "PP",
}
_LOAD_FIELDS = (
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
)


def scaling_power(pattern: str, term: str) -> int:
    return SCALING_POWER.get(pattern, {}).get(term, 0)


@dataclass(frozen=True)
class StageLoad:
    """一次阶段实验的负载。标准轴不够时放进 extras。

    length 是自注意力的填充长度。n_real 是这一批真实 token 总数，
    线性投影按它算。各样本等长时 n_real = batch · length。
    """

    batch: int
    length: int = 512
    n_real: int | None = None
    n_id: int = 128
    n_query: int = 8
    length_kv: int = 512
    tokens: int = 16
    groups: int = 4
    width: int = 64
    hidden: int = 256
    n_head: int = 4
    d_head: int = 16
    extras: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.n_real is None:
            object.__setattr__(self, "n_real", self.batch * self.length)
        object.__setattr__(self, "extras", dict(self.extras or {}))
        named = {field: getattr(self, field) for field in _LOAD_FIELDS}
        named.update(self.extras)
        for name, value in named.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise FrameworkError(f"{name} 至少为 1")

    def varied(self, **kwargs: int | Mapping[str, int]) -> "StageLoad":
        extras = dict(self.extras)
        if "extras" in kwargs:
            extras.update(kwargs.pop("extras"))  # type: ignore[arg-type]
        values = {field: getattr(self, field) for field in _LOAD_FIELDS}
        values.update(kwargs)
        if "n_real" not in kwargs and ("batch" in kwargs or "length" in kwargs):
            values["n_real"] = int(values["batch"]) * int(values["length"])
        values["extras"] = extras
        return StageLoad(**values)


def default_load() -> StageLoad:
    return StageLoad(batch=2)


@dataclass(frozen=True)
class StageSpec:
    """结构里的一个阶段。axes 把算子的轴改接到别的字段，包括 extras。"""

    name: str
    operator: str
    n_layers: int = 1
    elem_bytes: int = 2
    token_axis: str = "real"
    groups_follow_length: bool = False
    axes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise FrameworkError(f"未知的算子 {self.operator}")
        if self.token_axis not in TOKEN_AXES:
            raise FrameworkError(f"{self.name} 的 token 轴未知")
        if self.n_layers < 1 or self.elem_bytes < 1:
            raise FrameworkError(f"{self.name} 的层数和元素字节数至少为 1")
        if not self.name:
            raise FrameworkError("阶段名称不能为空")
        object.__setattr__(self, "axes", dict(self.axes or {}))
        for role, field in self.axes.items():
            if not role or not field:
                raise FrameworkError(f"{self.name} 的轴绑定不完整")


@dataclass(frozen=True)
class StageSplit:
    kind: str
    degree: int = 1
    n_microbatches: int = 1

    def __post_init__(self) -> None:
        if self.kind not in SPLIT_PATTERNS:
            raise FrameworkError(f"未知的切分 {self.kind}")
        if self.degree < 1 or self.n_microbatches < 1:
            raise FrameworkError("并行度和 micro-batch 数至少为 1")
        if self.kind == "none" and (self.degree != 1 or self.n_microbatches != 1):
            raise FrameworkError("不切分的并行度为 1")
        if self.kind != "none" and self.degree < 2:
            raise FrameworkError(f"{self.kind} 的并行度至少为 2")
        if self.kind != "pp" and self.n_microbatches != 1:
            raise FrameworkError("只有流水并行记录 micro-batch 数")


def ensure_legal(spec: StageSpec, split: StageSplit) -> None:
    if split.kind not in LEGAL_SPLITS[spec.operator]:
        raise FrameworkError(
            f"{spec.name} 是{OPERATOR_LABEL[spec.operator]}，"
            f"不能用 {SPLIT_LABEL[split.kind]}"
        )


def axis(spec: StageSpec, load: StageLoad, role: str, default: str) -> float:
    field = spec.axes.get(role, default)
    if field in _LOAD_FIELDS:
        return float(getattr(load, field))
    if field not in load.extras:
        raise FrameworkError(f"{spec.name} 要读轴 {field}，负载里没有")
    return float(load.extras[field])


def groups_of(spec: StageSpec, load: StageLoad) -> float:
    if spec.groups_follow_length:
        tokens = axis(spec, load, "tokens", "tokens")
        length = axis(spec, load, "length", "length")
        return max(1.0, length // tokens)
    return axis(spec, load, "groups", "groups")


def linear_tokens(spec: StageSpec, load: StageLoad) -> float:
    if spec.token_axis == "real":
        return axis(spec, load, "n_real", "n_real")
    if spec.token_axis == "group":
        return load.batch * groups_of(spec, load) * axis(spec, load, "tokens", "tokens")
    if spec.token_axis == "fixed":
        return load.batch * axis(spec, load, "tokens", "tokens")
    return load.batch * axis(spec, load, "n_query", "n_query")


def features(spec: StageSpec, load: StageLoad) -> dict[str, float]:
    """候选单项。拟合时只保留能降低相对误差的那些。"""
    batch = float(load.batch)
    width = axis(spec, load, "width", "width")
    groups = groups_of(spec, load)
    tokens = axis(spec, load, "tokens", "tokens")
    length = axis(spec, load, "length", "length")
    return {
        "lookup": batch * axis(spec, load, "n_id", "n_id"),
        "linear": linear_tokens(spec, load) * width * axis(spec, load, "hidden", "hidden"),
        "self_attn": (
            batch
            * axis(spec, load, "n_head", "n_head")
            * length
            * length
            * axis(spec, load, "d_head", "d_head")
        ),
        "attn_linear": (
            batch
            * axis(spec, load, "n_head", "n_head")
            * length
            * axis(spec, load, "d_head", "d_head")
        ),
        "cross_attn": (
            batch
            * axis(spec, load, "n_query", "n_query")
            * axis(spec, load, "length_kv", "length_kv")
            * width
        ),
        "token_square": batch * groups * tokens * tokens * width,
        "token_linear": batch * groups * tokens * width,
        "permute": batch * tokens * width,
        "intercept": 1.0,
    }


def volume_bytes(spec: StageSpec, load: StageLoad, split: StageSplit) -> float:
    """每卡集合通信字节数。shard_r 组间没有归约，体积为 0。

    流水的 micro-batch 把同一批 token 切开再送出，总字节不随 micro-batch 数变。
    """
    if split.kind != "none":
        ensure_legal(spec, split)
    if split.kind in ("none", "shard_r"):
        return 0.0
    degree = float(split.degree)
    per_rank = (degree - 1.0) / degree
    layers = float(spec.n_layers)
    elem = float(spec.elem_bytes)
    batch = float(load.batch)
    width = axis(spec, load, "width", "width")
    length = axis(spec, load, "length", "length")
    if split.kind == "cp_q":
        return per_rank * layers * batch * length * width * elem
    if split.kind == "cp_qk":
        heads = axis(spec, load, "n_head", "n_head")
        d_head = axis(spec, load, "d_head", "d_head")
        return 2.0 * per_rank * layers * batch * length * heads * d_head * elem
    if split.kind == "shard_kv":
        queries = axis(spec, load, "n_query", "n_query")
        return 2.0 * per_rank * layers * batch * queries * width * elem
    if split.kind == "shard_q":
        queries = axis(spec, load, "n_query", "n_query")
        return per_rank * layers * batch * queries * width * elem
    if split.kind == "tp":
        return 2.0 * per_rank * layers * linear_tokens(spec, load) * width * elem
    if split.kind == "embp":
        ids = axis(spec, load, "n_id", "n_id")
        return per_rank / degree * layers * batch * ids * width * elem
    if split.kind == "shard_t":
        tokens = axis(spec, load, "tokens", "tokens")
        if spec.operator == "token_permute":
            count = batch * tokens
        else:
            count = batch * groups_of(spec, load) * tokens
        return per_rank * layers * count * width * elem
    if split.kind == "pp":
        return (degree - 1.0) * linear_tokens(spec, load) * width * elem
    raise FrameworkError(f"{split.kind} 没有通信体积")


def compute_parts(
    spec: StageSpec,
    coefficients: Mapping[str, float],
    load: StageLoad,
    split: StageSplit,
    kappa: float,
) -> tuple[float, float, float]:
    """返回计算时间、κ 的特征、不含 κ 的偏移。

    计算时间 = 偏移 + κ · κ特征。κ = 0 且 power = 1 时，被切项变成原来的 1/N。
    """
    if kappa < 0.0:
        raise FrameworkError("κ 不能为负")
    feats = features(spec, load)
    if split.kind == "none":
        total = sum(coefficients.get(name, 0.0) * feats[name] for name in FEATURE_ORDER)
        return total, 0.0, total
    ensure_legal(spec, split)
    offset = 0.0
    kappa_feature = 0.0
    bubble = pipeline_factor(split.degree, split.n_microbatches)
    degree = float(split.degree)
    for name in FEATURE_ORDER:
        value = coefficients.get(name, 0.0) * feats[name]
        if split.kind == "pp":
            offset += value * bubble
            continue
        power = scaling_power(split.kind, name)
        if power == 0:
            offset += value
            continue
        offset += value / degree**power
        kappa_feature += value * (degree - 1.0) / degree**power
    return offset + kappa * kappa_feature, kappa_feature, offset
