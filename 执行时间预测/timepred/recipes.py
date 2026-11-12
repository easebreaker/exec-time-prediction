"""四个模型各自拆成哪些阶段。

测量时一个负载要同时带上各阶段读的轴。HyFormer 的两段交叉注意力
读的 query 长度不同，用 axes 接到 extras 里的不同字段。
"""

from __future__ import annotations

from .schema import FrameworkError
from .stage import StageSpec


def hstu_stages(n_layers: int = 1) -> tuple[StageSpec, ...]:
    """查表、UVQK 投影、自注意力。投影按真实 token 数，分数按填充长度。"""
    return (
        StageSpec("embed", "lookup"),
        StageSpec("proj", "linear", n_layers=n_layers),
        StageSpec("attn", "self_attn", n_layers=n_layers),
    )


def mtgr_stages(n_layers: int = 1) -> tuple[StageSpec, ...]:
    """与 HSTU 同一组算子。

    交叉特征写在候选 token 里，不单列交叉注意力。画像、历史、实时行为和候选
    拼成一条序列后做自注意力。掩码让候选之间互不可见；稠密实现仍会形成
    整段 L² 的分数，能跳过掩码块的 kernel 会在复杂度实验里表现为更低的次数。
    """
    return hstu_stages(n_layers)


def longer_stages() -> tuple[StageSpec, ...]:
    """组内注意力的组数由 L/T 推出，因此随用户序列长度变。

    短序列上的自注意力读 extras['short_len']，不读原始长度。
    混合交叉注意力的 N_q 和 L_kv 用负载上的 n_query、length_kv。
    """
    return (
        StageSpec("embed", "lookup"),
        StageSpec("inner_attn", "group_attn", groups_follow_length=True),
        StageSpec(
            "inner_ffn",
            "linear",
            token_axis="group",
            groups_follow_length=True,
        ),
        StageSpec("hybrid_cross", "cross_attn"),
        StageSpec("short_attn", "self_attn", axes={"length": "short_len"}),
    )


def hyformer_stages(encoder: str = "longer") -> tuple[StageSpec, ...]:
    """Query Decoding 是交叉注意力，Query Boosting 是转置混合加逐 token 投影。

    decode 的 query 数和 KV 长度都是 short_len，不读用户序列 length。
    序列侧交叉注意力的 query 数也是 short_len，KV 长度是 extras['kv_len']。
    kv_len 与 length、short_len 都不是同一列，三条轴可以分开扫。
    Boosting 的 token 数放在 extras['boost_tokens']，不读用户序列长度。
    """
    if encoder not in ("longer", "full", "swiglu"):
        raise FrameworkError(f"未知的 HyFormer 编码器 {encoder}")
    stages: list[StageSpec] = [StageSpec("embed", "lookup")]
    if encoder == "full":
        stages.append(StageSpec("seq_attn", "self_attn"))
        stages.append(StageSpec("seq_ffn", "linear"))
    elif encoder == "longer":
        stages.append(
            StageSpec(
                "seq_cross",
                "cross_attn",
                axes={"n_query": "short_len", "length_kv": "kv_len"},
            )
        )
        stages.append(StageSpec("seq_ffn", "linear", token_axis="query", axes={"n_query": "short_len"}))
    else:
        stages.append(StageSpec("seq_ffn", "linear"))
    stages.extend(
        (
            StageSpec(
                "decode",
                "cross_attn",
                axes={"n_query": "short_len", "length_kv": "short_len"},
            ),
            StageSpec("boost_mix", "token_permute", axes={"tokens": "boost_tokens"}),
            StageSpec(
                "boost_ffn",
                "linear",
                token_axis="fixed",
                axes={"tokens": "boost_tokens"},
            ),
        )
    )
    return tuple(stages)


def rankmixer_stages() -> tuple[StageSpec, ...]:
    """转置混合没有 T²。逐 token FFN 的长度是固定的 T。"""
    return (
        StageSpec("mix", "token_permute"),
        StageSpec("ffn", "linear", token_axis="fixed"),
    )


def token_mixer_stage(groups_follow_length: bool = False) -> StageSpec:
    """token 轴上的矩阵乘。组数不随 L 变时，时间和用户序列长度无关。"""
    return StageSpec(
        "token_mixer",
        "token_gemm",
        groups_follow_length=groups_follow_length,
    )


def hstu_stages_with_elemwise(n_layers: int = 1) -> tuple[StageSpec, ...]:
    """把 HSTU 一层拆成「投影 → 归一化 → 注意力 → 激活 → 归一化」。

    与 `hstu_stages` 数值等价的前提是归一化段 fused=True（被融合进相邻 kernel，
    整段为 0）。设 fused=False 时这些段开始占时间，用来回答「融合没生效会多多少」。
    """
    return (
        StageSpec("embed", "lookup", n_layers=n_layers),
        StageSpec("proj", "linear", n_layers=n_layers),
        StageSpec("norm_in", "elemwise", n_layers=n_layers, fused=True),
        StageSpec("attn", "self_attn", n_layers=n_layers),
        StageSpec("act", "elemwise", n_layers=n_layers, fused=True),
        StageSpec("norm_out", "elemwise", n_layers=n_layers, fused=True),
    )


def hstu_stages_tiled(n_layers: int = 1, block: int = 128) -> tuple[StageSpec, ...]:
    """自注意力改用 tile 计数（块数），替代 L 的幂。块大小由实现登记。"""
    return (
        StageSpec("embed", "lookup", n_layers=n_layers),
        StageSpec("proj", "linear", n_layers=n_layers),
        StageSpec(
            "attn",
            "self_attn",
            n_layers=n_layers,
            model="tile",
            block_m=block,
            block_n=block,
        ),
    )


def moe_ffn_stages(
    n_layers: int = 1,
    n_exp: int = 64,
    topk: int = 6,
    renorm: int = 0,
    span: str = "intra",
) -> tuple[StageSpec, ...]:
    """MoE 的一层：门控 → 分发 → 专家 FFN → 合并 → 逐元素。

    分发的通信量走 ep 切分（`StageSplit("ep", degree, span=span)`）的 τ + v·V，
    本段只记路由与搬运。`span` 决定用机内还是跨机的每字节代价。
    """
    return (
        StageSpec("gate", "moe_gate", n_layers=n_layers, renorm=renorm),
        StageSpec("dispatch", "moe_comm", n_layers=n_layers),
        StageSpec("expert_ffn", "linear", n_layers=n_layers),
        StageSpec("combine", "moe_comm", n_layers=n_layers),
        StageSpec("act", "elemwise", n_layers=n_layers, fused=True),
    )


def retrieval_stages(n_layers: int = 1) -> tuple[StageSpec, ...]:
    """召回一步：用户向量与全库向量打分。N_item 是主项，不是 L。"""
    return (
        StageSpec("user_tower", "linear", n_layers=n_layers, token_axis="query"),
        StageSpec("score", "score_gemm", n_layers=n_layers),
    )
