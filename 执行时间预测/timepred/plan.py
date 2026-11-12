"""四个模型在现成权重上要跑的请求。

复杂度网格里的宽度、头数不在这里。这里只改输入，以及 MTGR 已经有的 CP。
交叉注意力的 KV 长度不写进用户序列 length。
"""

from __future__ import annotations

from dataclasses import dataclass

# 拟合用的用户序列长度。16384 留出，不进拟合。
FIT_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
HOLDOUT_LENGTH = 16384
# 只在一个长度上改并行度时，κ 的特征和体积都正比于 (N-1)/N，拟合会把 κ、v 收成 0，只留下 τ。
CP_DEGREES = (2, 4, 8)
CP_LENGTHS = (512, 1024, 2048, 4096, 8192)
BATCHES = (2, 4, 8)
BATCH_LENGTHS = (512, 2048)
GROUP_TOKENS = 16
SHORT_LEN = 64
KV_LEN = 512
BOOST_TOKENS = 16
SHORT_SWEEP = (16, 32, 128, 256)
KV_SWEEP = (128, 256, 1024, 2048, 4096, 8192)
BOOST_SWEEP = (4, 8, 32, 64)


@dataclass(frozen=True)
class MeasureRequest:
    model: str
    id: str
    series: str
    role: str
    batch: int
    length: int
    n_real: int
    n_id: int
    tokens: int
    groups: int
    n_query: int
    length_kv: int
    short_len: int | None
    kv_len: int | None
    boost_tokens: int | None
    split: str
    degree: int
    note: str
    # 补登记的轴。旧四模型不用，MoE 与召回的请求才填。
    n_exp: int | None = None
    topk: int | None = None
    n_item: int | None = None


def _row(
    model: str,
    request_id: str,
    series: str,
    role: str,
    *,
    batch: int = 1,
    length: int = 512,
    tokens: int = GROUP_TOKENS,
    groups: int | None = None,
    n_id: int | None = None,
    short_len: int | None = None,
    kv_len: int | None = None,
    boost_tokens: int | None = None,
    split: str = "none",
    degree: int = 1,
    note: str,
    n_exp: int | None = None,
    topk: int | None = None,
    n_item: int | None = None,
) -> MeasureRequest:
    n_real = batch * length
    return MeasureRequest(
        model=model,
        id=request_id,
        series=series,
        role=role,
        batch=batch,
        length=length,
        n_real=n_real,
        # n_id 是每样本的查表次数：stage.py 的 lookup 主项是 B·n_id，
        # 再乘 batch 会把 total 重复算一次。样本等长且无填充时即 length。
        n_id=(n_real // batch) if n_id is None else n_id,
        tokens=tokens,
        groups=4 if groups is None else groups,
        n_query=8,
        length_kv=KV_LEN,
        short_len=short_len,
        kv_len=kv_len,
        boost_tokens=boost_tokens,
        split=split,
        degree=degree,
        note=note,
        n_exp=n_exp,
        topk=topk,
        n_item=n_item,
    )


def _length_and_batch(model: str, prefix: str, *, longer: bool) -> list[MeasureRequest]:
    rows: list[MeasureRequest] = []
    length_note = "稠密。n_id 预填为每样本次数（等于 length），以实际查表次数为准。"
    if longer:
        length_note += " R=⌊L/T⌋，T 与权重不符则整列替换并重算 R。"
    for length in FIT_LENGTHS:
        groups = max(1, length // GROUP_TOKENS) if longer else None
        rows.append(
            _row(
                model,
                f"{prefix}-L{length:05d}",
                "长度",
                "拟合",
                length=length,
                groups=groups,
                short_len=SHORT_LEN if longer else None,
                note=length_note,
            )
        )
    hold_groups = max(1, HOLDOUT_LENGTH // GROUP_TOKENS) if longer else None
    rows.append(
        _row(
            model,
            f"{prefix}-L{HOLDOUT_LENGTH:05d}",
            "留出",
            "留出",
            length=HOLDOUT_LENGTH,
            groups=hold_groups,
            short_len=SHORT_LEN if longer else None,
            note="长度超出拟合范围。不进拟合。",
        )
    )
    for length in BATCH_LENGTHS:
        groups = max(1, length // GROUP_TOKENS) if longer else None
        for batch in BATCHES:
            rows.append(
                _row(
                    model,
                    f"{prefix}-B{batch:02d}-L{length}",
                    "batch",
                    "拟合",
                    batch=batch,
                    length=length,
                    groups=groups,
                    short_len=SHORT_LEN if longer else None,
                    note="这一档长度上只改 batch。n_real=B·L。",
                )
            )
    return rows


def hstu_requests() -> tuple[MeasureRequest, ...]:
    """与 MTGR 同一组算子。长度和 batch 都要有，用来看掩码把次数拉低多少。"""
    return tuple(_length_and_batch("HSTU", "HS", longer=False))


def longer_requests() -> tuple[MeasureRequest, ...]:
    """用户长度、batch，再加上 short_len。混合交叉的 N_q、L_kv 不在这批里预填。"""
    rows = _length_and_batch("LONGER", "LG", longer=True)
    for short in SHORT_SWEEP:
        rows.append(
            _row(
                "LONGER",
                f"LG-S{short:04d}",
                "短序列",
                "拟合",
                groups=max(1, 512 // GROUP_TOKENS),
                short_len=short,
                note="用户 L 固定。主要读 short_attn。",
            )
        )
    return tuple(rows)


def mtgr_requests() -> tuple[MeasureRequest, ...]:
    """长度、batch，以及 CP 的并行度×长度。只切 Query，通信两列必填。"""
    rows = _length_and_batch("MTGR", "MT", longer=False)
    for degree in CP_DEGREES:
        for length in CP_LENGTHS:
            rows.append(
                _row(
                    "MTGR",
                    f"MT-C{degree:02d}-L{length}",
                    "切分",
                    "切分",
                    length=length,
                    split="cp_q",
                    degree=degree,
                    note="只切 Query，只记 attn。通信字节和通信时间必填。与同长度的未切分请求一次拟合。",
                )
            )
    return tuple(rows)


def hyformer_requests() -> tuple[MeasureRequest, ...]:
    """decode 的 Q 与 KV 都是 short_len。序列侧 KV 是 kv_len，不读用户序列 length。"""
    rows: list[MeasureRequest] = []
    for length in FIT_LENGTHS:
        rows.append(
            _row(
                "HyFormer",
                f"HF-L{length:05d}",
                "长度",
                "拟合",
                length=length,
                short_len=SHORT_LEN,
                kv_len=KV_LEN,
                boost_tokens=BOOST_TOKENS,
                note="short_len 与 kv_len 不动。seq_cross 和 decode 不应随用户 L 变化。",
            )
        )
    rows.append(
        _row(
            "HyFormer",
            f"HF-L{HOLDOUT_LENGTH:05d}",
            "留出",
            "留出",
            length=HOLDOUT_LENGTH,
            short_len=SHORT_LEN,
            kv_len=KV_LEN,
            boost_tokens=BOOST_TOKENS,
            note="用户长度超出拟合范围。不进拟合。",
        )
    )
    for short in SHORT_SWEEP:
        rows.append(
            _row(
                "HyFormer",
                f"HF-S{short:04d}",
                "短序列",
                "拟合",
                short_len=short,
                kv_len=KV_LEN,
                boost_tokens=BOOST_TOKENS,
                n_id=512,
                note="decode 的 Q 和 KV 都是 short_len。seq_cross 只改 query 一侧。",
            )
        )
    for kv in KV_SWEEP:
        rows.append(
            _row(
                "HyFormer",
                f"HF-K{kv:05d}",
                "KV",
                "拟合",
                short_len=SHORT_LEN,
                kv_len=kv,
                boost_tokens=BOOST_TOKENS,
                n_id=512,
                note="只改 kv_len。decode 不读这一列，seq_cross 的 KV 读它。",
            )
        )
    for boost in BOOST_SWEEP:
        rows.append(
            _row(
                "HyFormer",
                f"HF-B{boost:04d}",
                "boost",
                "拟合",
                short_len=SHORT_LEN,
                kv_len=KV_LEN,
                boost_tokens=boost,
                n_id=512,
                note="n_id 钉在 512。实际查表次数变了就改，并补一次查表次数不变的构造。",
            )
        )
    rows.append(
        _row(
            "HyFormer",
            "HF-J-S128-K2048",
            "留出",
            "留出",
            short_len=128,
            kv_len=2048,
            boost_tokens=BOOST_TOKENS,
            n_id=512,
            note="short_len 与 kv_len 同时离开单轴。不进拟合。",
        )
    )
    return tuple(rows)


def all_requests() -> tuple[MeasureRequest, ...]:
    return mtgr_requests() + hstu_requests() + longer_requests() + hyformer_requests()


# ---------------------------------------------------------------- 补登记的轴
# 下面两类请求**不进 all_requests()**：它们要挂在含 MoE 或含召回的模型上，
# 而现成的四个权重里没有这两段。原 88 条保持不变，这里只是把「该测什么」写出来。

EXP_SWEEP = (32, 64, 128, 256)
EP_DEGREES = (2, 4, 8)
EP_SPANS = ("intra", "cross")
ITEM_SWEEP = (128, 1024, 8192, 65536)


def moe_requests(model: str = "MoE-参考", length: int = 512) -> tuple[MeasureRequest, ...]:
    """门控与分发的测定计划。

    门控：只改专家数，验证成本随 N_exp 线性、且 renorm=1 时打分只对 top-k 做。
    分发：只改 EP 并行度与跨度，验证 V 随 topk·token 数变、v 分两档。
    """
    rows: list[MeasureRequest] = []
    for n_exp in EXP_SWEEP:
        rows.append(
            _row(
                model,
                f"MO-G-N{n_exp:04d}",
                "专家数",
                "拟合",
                length=length,
                n_exp=n_exp,
                topk=6,
                note="门控。只改 N_exp，token 数不变。",
            )
        )
    for degree in EP_DEGREES:
        for span in EP_SPANS:
            rows.append(
                _row(
                    model,
                    f"MO-D-E{degree}-{span[0].upper()}",
                    "专家并行",
                    "拟合",
                    length=length,
                    n_exp=64,
                    topk=6,
                    split="ep",
                    degree=degree,
                    note=f"分发。EP={degree}，跨度={span}；每档各一组，用于分开两档 v。",
                )
            )
    rows.append(
        _row(
            model,
            "MO-D-E8-C-L2048",
            "专家并行",
            "拟合",
            length=2048,
            n_exp=64,
            topk=6,
            split="ep",
            degree=8,
            note="同一 EP、换长度：让 V 与 token 数分离，否则与 κ 共线。",
        )
    )
    return tuple(rows)


def retrieval_requests(model: str = "召回-参考", length: int = 64) -> tuple[MeasureRequest, ...]:
    """全库打分：只改商品数，验证成本随 N_item 线性且与长度无关。"""
    rows: list[MeasureRequest] = []
    for n_item in ITEM_SWEEP:
        rows.append(
            _row(
                model,
                f"RT-S-I{n_item:06d}",
                "商品数",
                "拟合" if n_item < ITEM_SWEEP[-1] else "留出",
                length=length,
                n_item=n_item,
                note="全库打分。用户长度固定，只改商品数。",
            )
        )
    return tuple(rows)
