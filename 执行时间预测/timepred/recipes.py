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

    encoder 为 longer 时，序列侧先做短 query 对长 KV 的交叉注意力。
    短序列长度放在 extras['short_len']，解码 query 数放在 extras['decode_queries']。
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
                axes={"n_query": "short_len", "length_kv": "length"},
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
                axes={"n_query": "decode_queries", "length_kv": "short_len"},
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
