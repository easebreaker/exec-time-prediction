"""一个模型写成阶段序列。每个阶段只做一种算子。

整次前向的时间是各阶段时间之和。复杂度、切分加速比和通信按阶段估计，
不按 HyFormer、LONGER、HSTU、MTGR 的名字估计。这些模型的差别是阶段
怎么排列，以及长度轴接到哪一个输入。

算子和它在单卡上的主项：

- lookup 查表：B · n_id
- linear 线性投影：N_tok · d · d_hid。N_tok 由 token_axis 决定。
- self_attn 自注意力：B · n_head · L² · d_head。L 是该批的填充长度。
  sdpa=True 时改用融合内核：平方项返回 0，只留一次项
  attn_linear = B · n_head · L · d_head。默认 sdpa=False，一次项返回 0。
  切分仍只缩小平方项。SDPA 的一次项还要读整段 KV，不除以并行度。
- cross_attn 交叉注意力：B · N_q · L_kv · d。L_kv 是独立参数，读 length_kv，
  或用 axes 接到 short_len、kv_len 等字段。不读用户序列 length。
  HyFormer 的 decode 上，query 数和 KV 长度都是 short_len。
  序列侧交叉注意力的 KV 接到 kv_len，与 short_len、length 分开。
- group_attn 组内注意力：B · R · T² · d。groups_follow_length 时 R = L / T。
- token_gemm token 轴矩阵乘：与组内注意力同一单项。R 固定时不读 L。
- token_permute 转置混合：B · T · d，按字节计，没有 T²。

补登记的四类（见 `文档/算子复杂度-按算子手册.md` §8–§11）：

- elemwise 归一化与逐元素：linear 项按元素数算，square 项给「对 L² 分数矩阵做
  逐元素」的情形（HSTU 对 S×S 做 SiLU）。fused=True 时整段为 0。
- moe_gate 门控与 top-k：B · L · N_exp 的打分，加一项排序。renorm=1 时打分
  只对 top-k 个值做，cost 从 N_exp 降到 k。
- moe_comm 专家分发与合并：本算子只做路由与搬运，通信走 ep 切分的 τ + v·V。
- score_gemm 全库打分：N_user · N_item · d，M 极小时为带宽 bound。

切分只缩小它切得到的那一项。自注意力只切 Query、KV 留整段时，平方项除以
并行度；Query 和 KV 都切开时，平方项除以并行度的平方。线性投影按隐藏维切
时除以并行度。按组切开且组之间没有归约时，没有集合通信。

登记位（不可拟合的事实，全部由实现决定）：

- impl：实现路径 A/B/C，含义见手册各章。
- fused：是否已被相邻 kernel 吸收。
- sparse：块级跳过后的有效比例 σ（causal 0.5、滑窗 w/L、块对角 1/T）。
- block_m / block_n：基本块；model="tile" 时用它替代 L 的幂次。
- model：power（现状）或 tile（tile 计数）。
- intercept_mode：stage（每段一个截距）或 network（截距提到网络层）。
- comm_span：通信跨度，intra（机内）或 cross（跨机）；v 可以按跨度不同（记在 StageSplit）。
- value_dim：注意力的 V 维；与 d_head 不同时（HSTU 的 linear_dim），L² 项宽度取二者之和。
- kernels_per_layer：每层下发几个 kernel，网络层固定开销的唯一自变量。
"""

from __future__ import annotations

import math
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
    "elemwise",
    "moe_gate",
    "moe_comm",
    "score_gemm",
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
    "ep",
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
    "elemwise_linear",
    "elemwise_square",
    "gate_score",
    "gate_sort",
    "moe_route",
    "score",
    "tile",
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
    "elemwise": "elemwise_linear",
    "moe_gate": "gate_score",
    "moe_comm": "moe_route",
    "score_gemm": "score",
}
LEGAL_SPLITS = {
    "lookup": ("none", "embp"),
    "linear": ("none", "tp", "pp"),
    "self_attn": ("none", "cp_q", "cp_qk"),
    "cross_attn": ("none", "shard_kv", "shard_q"),
    "group_attn": ("none", "shard_r"),
    "token_gemm": ("none", "shard_r", "shard_t"),
    "token_permute": ("none", "shard_t"),
    "elemwise": ("none",),
    "moe_gate": ("none",),
    "moe_comm": ("none", "ep"),
    "score_gemm": ("none",),
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
    # 专家并行只换位，不改本段计算量；通信单独记在 volume_bytes。
    "ep": {},
}
OPERATOR_LABEL = {
    "lookup": "查表",
    "linear": "线性投影",
    "self_attn": "自注意力",
    "cross_attn": "交叉注意力",
    "group_attn": "组内注意力",
    "token_gemm": "token 轴矩阵乘",
    "token_permute": "转置混合",
    "elemwise": "归一化与逐元素",
    "moe_gate": "门控与 top-k",
    "moe_comm": "专家分发与合并",
    "score_gemm": "全库打分",
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
    "ep": "EP（专家并行）",
}
IMPL_PATHS = ("auto", "A", "B", "C")
COMPLEXITY_MODELS = ("power", "tile")
INTERCEPT_MODES = ("stage", "network")
COMM_SPANS = ("intra", "cross")
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
    "n_exp",
    "topk",
    "n_user",
    "n_item",
    "n_probe",
)


def scaling_power(pattern: str, term: str) -> int:
    return SCALING_POWER.get(pattern, {}).get(term, 0)


def primary_term(spec: "StageSpec") -> str:
    """这一阶段在拟合里要保住的单项。SDPA 的主项是一次项，不是 L²。"""
    if spec.model == "tile" and spec.operator in ("self_attn", "cross_attn"):
        return "tile"
    if spec.sdpa:
        return "attn_linear"
    return PRIMARY_TERM[spec.operator]


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
    # 补登记的两条轴：专家数与商品数。二者的成本都不是 L 的幂。
    n_exp: int = 64
    topk: int = 6
    n_user: int = 1
    n_item: int = 1000
    n_probe: int = 0
    extras: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if self.n_real is None:
            object.__setattr__(self, "n_real", self.batch * self.length)
        object.__setattr__(self, "extras", dict(self.extras or {}))
        named = {field: getattr(self, field) for field in _LOAD_FIELDS}
        named.update(self.extras)
        for name, value in named.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FrameworkError(f"{name} 至少为 0")

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
    """结构里的一个阶段。axes 把算子的轴改接到别的字段，包括 extras。

    后七个字段是登记位（实现事实），不是拟合参数：
    impl / fused / sparse / block_m / block_n / model / intercept_mode。
    """

    name: str
    operator: str
    n_layers: int = 1
    elem_bytes: int = 2
    token_axis: str = "real"
    groups_follow_length: bool = False
    sdpa: bool = False
    axes: Mapping[str, str] | None = None
    impl: str = "auto"
    fused: bool = False
    sparse: float = 1.0
    block_m: int = 0
    block_n: int = 0
    model: str = "power"
    intercept_mode: str = "stage"
    # moe_gate 的登记位：0 = 先 softmax 再 top-k；1 = 先 top-k 再 softmax。
    renorm: int = 0
    # 每个阶段每层下发几个 kernel。网络层固定开销 = Σ n_layers × 这个数。
    kernels_per_layer: int = 1
    # 注意力的 V 维。为 0 表示与 d_head 相同；HSTU 的 linear_dim ≠ attention_dim 时
    # L² 项的宽度是二者之和（p·v 的收缩维是 V，不是 d_head）。
    value_dim: int = 0

    def __post_init__(self) -> None:
        if self.operator not in OPERATORS:
            raise FrameworkError(f"未知的算子 {self.operator}")
        if self.sdpa and self.operator != "self_attn":
            raise FrameworkError(f"{self.name} 不是自注意力，不能开 SDPA")
        if self.token_axis not in TOKEN_AXES:
            raise FrameworkError(f"{self.name} 的 token 轴未知")
        if self.n_layers < 1 or self.elem_bytes < 1:
            raise FrameworkError(f"{self.name} 的层数和元素字节数至少为 1")
        if not self.name:
            raise FrameworkError("阶段名称不能为空")
        if self.impl not in IMPL_PATHS:
            raise FrameworkError(f"{self.name} 的实现路径只能是 {IMPL_PATHS}")
        if self.model not in COMPLEXITY_MODELS:
            raise FrameworkError(f"{self.name} 的复杂度模型只能是 {COMPLEXITY_MODELS}")
        if self.intercept_mode not in INTERCEPT_MODES:
            raise FrameworkError(f"{self.name} 的截距模式只能是 {INTERCEPT_MODES}")
        if not 0.0 < self.sparse <= 1.0:
            raise FrameworkError(f"{self.name} 的稀疏度要在 (0, 1] 内")
        if self.model == "tile" and (self.block_m < 1 or self.block_n < 1):
            raise FrameworkError(f"{self.name} 用 tile 模型时 block_m、block_n 至少为 1")
        if self.block_m < 0 or self.block_n < 0:
            raise FrameworkError(f"{self.name} 的基本块不能为负")
        if self.renorm not in (0, 1):
            raise FrameworkError(f"{self.name} 的 renorm 只能是 0 或 1")
        if self.kernels_per_layer < 0:
            raise FrameworkError(f"{self.name} 的每层 kernel 数不能为负")
        if self.value_dim < 0:
            raise FrameworkError(f"{self.name} 的 value_dim 不能为负")
        if self.fused and self.operator not in (
            "elemwise",
            "token_permute",
            "lookup",
            "linear",
        ):
            raise FrameworkError(f"{self.name} 这种算子不会被相邻 kernel 吸收")
        object.__setattr__(self, "axes", dict(self.axes or {}))
        for role, field in self.axes.items():
            if not role or not field:
                raise FrameworkError(f"{self.name} 的轴绑定不完整")
        if self.operator == "cross_attn" and self.axes.get("length_kv") == "length":
            raise FrameworkError(
                f"{self.name} 的 KV 长度不要接到用户序列 length 上，用 length_kv 或单独的轴"
            )


@dataclass(frozen=True)
class StageSplit:
    kind: str
    degree: int = 1
    n_microbatches: int = 1
    # 通信跨度。v 可按跨度不同：机内与跨机差约一个量级。
    span: str = "intra"

    def __post_init__(self) -> None:
        if self.kind not in SPLIT_PATTERNS:
            raise FrameworkError(f"未知的切分 {self.kind}")
        if self.span not in COMM_SPANS:
            raise FrameworkError(f"通信跨度只能是 {COMM_SPANS}")
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


def tile_count(spec: StageSpec, load: StageLoad) -> float:
    """按基本块数计的张量块数（含头与 batch）。

    self_attn 的 Query 侧被并行度切开，KV 侧留整段；cross_attn 的 Query 侧
    是 N_q、KV 侧是 L_kv。块数对 L 是台阶函数，不是幂。
    """
    if spec.block_m < 1 or spec.block_n < 1:
        return 0.0
    batch = float(load.batch)
    heads = axis(spec, load, "n_head", "n_head")
    if spec.operator == "self_attn":
        rows = axis(spec, load, "length", "length")
        cols = rows
    elif spec.operator == "cross_attn":
        rows = axis(spec, load, "n_query", "n_query")
        cols = axis(spec, load, "length_kv", "length_kv")
    else:
        return 0.0
    return batch * heads * math.ceil(rows / spec.block_m) * math.ceil(cols / spec.block_n)


def features(spec: StageSpec, load: StageLoad) -> dict[str, float]:
    """候选单项。拟合时只保留能降低相对误差的那些。

    登记位先进来：fused 的段整段为 0；sparse 乘在平方项上；model="tile"
    时把 L 的幂换成块数。这三处都会改变列的取值，但不会改变列的名字。
    """
    zero = {name: 0.0 for name in FEATURE_ORDER}
    if spec.fused:
        return zero

    batch = float(load.batch)
    width = axis(spec, load, "width", "width")
    groups = groups_of(spec, load)
    tokens = axis(spec, load, "tokens", "tokens")
    length = axis(spec, load, "length", "length")
    heads = axis(spec, load, "n_head", "n_head")
    d_head = axis(spec, load, "d_head", "d_head")
    topk = axis(spec, load, "topk", "topk")
    n_exp = axis(spec, load, "n_exp", "n_exp")
    op = spec.operator

    # 同一次拟合只走一种注意力实现。另一种公式返回 0，列不变，不会被选进模型。
    # L² 项的宽度是 d_head + value_dim：qkᵀ 收缩 d_head，p·v 收缩 V 维（HSTU 的 linear_dim）。
    width_l2 = d_head + float(spec.value_dim)
    tile = tile_count(spec, load)
    use_tile = spec.model == "tile" and tile > 0.0
    square = (
        0.0 if (spec.sdpa or use_tile) else batch * heads * length * length * width_l2 * spec.sparse
    )
    sdpa_term = 0.0 if use_tile else (batch * heads * length * d_head if spec.sdpa else 0.0)

    # 按算子只算相关特征，不相关的返回 0。
    values = zero
    values["lookup"] = batch * axis(spec, load, "n_id", "n_id") if op == "lookup" else 0.0
    values["linear"] = (
        linear_tokens(spec, load) * width * axis(spec, load, "hidden", "hidden")
        if op == "linear"
        else 0.0
    )
    values["self_attn"] = square if op == "self_attn" else 0.0
    values["attn_linear"] = sdpa_term if op == "self_attn" else 0.0
    # L_kv 与用户序列 length 是两列。接到 length 上会在改序列长度时把 KV 一起带走。
    values["cross_attn"] = (
        batch
        * axis(spec, load, "n_query", "n_query")
        * axis(spec, load, "length_kv", "length_kv")
        * width
        * spec.sparse
        if op == "cross_attn" and not use_tile
        else 0.0
    )
    values["token_square"] = (
        batch * groups * tokens * tokens * width
        if op in ("group_attn", "token_gemm")
        else 0.0
    )
    values["token_linear"] = (
        batch * groups * tokens * width if op in ("group_attn", "token_gemm") else 0.0
    )
    values["permute"] = batch * tokens * width if op == "token_permute" else 0.0
    values["tile"] = tile if use_tile else 0.0

    if op == "elemwise":
        # 读一遍写一遍，按元素数；square 项给「对 L² 分数矩阵做逐元素」的情形。
        values["elemwise_linear"] = 2.0 * linear_tokens(spec, load) * width
        values["elemwise_square"] = 2.0 * batch * heads * length * length * spec.sparse
    if op == "moe_gate":
        # renorm=1 时先取 top-k 再 softmax，打分只对 k 个值做。
        scored = topk if spec.renorm == 1 else n_exp
        values["gate_score"] = batch * length * scored * width
        values["gate_sort"] = batch * length * math.log2(max(n_exp, 2.0))
    if op == "moe_comm":
        values["moe_route"] = batch * length * topk
    if op == "score_gemm":
        items = axis(spec, load, "n_probe", "n_probe") or axis(spec, load, "n_item", "n_item")
        values["score"] = axis(spec, load, "n_user", "n_user") * items * width

    # 截距是每段的固定开销。提到网络层后这一段不再记。
    values["intercept"] = 0.0 if spec.intercept_mode == "network" else 1.0
    return values


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
    if split.kind == "ep":
        # 每个 token 发给 top-k 个专家，落到同号的概率 1/EP，其余跨号。
        topk = axis(spec, load, "topk", "topk")
        return per_rank * layers * batch * length * topk * width * elem
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
