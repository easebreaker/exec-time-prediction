"""用 Meta 官方的 HSTU 单层代码，核对框架算出的「次数」是否等于真实张量形状。

背景：`_vendor/generative-recommenders` 是 `meta-recsys/generative-recommenders`
的浅克隆（Apache-2.0）。它的一层 `SequentialTransductionUnitJagged` 依赖
`torch.ops.fbgemm` 的 jagged 算子，而 fbgemm_gpu 没有 macOS 轮子。本脚本把
两个 jagged 算子换成等价的 dense 实现，其余代码原样运行，因此核对的是
**官方实现里每个算子的真实形状**，不是我们自己写的复现版。

核对方式：在 proxy 里包一层 `mm`、`einsum`、`layer_norm`、`silu`，记录每次调用
的输入形状；再把记录换算成「乘加数 / 元素数」，与 `timepred.stage.features()`
给出的特征逐项相比。比值应为 1（同一量）或 2（读+写两遍的口径差）。

用法：
    git clone --depth 1 https://github.com/meta-recsys/generative-recommenders \
        _vendor/generative-recommenders          # 只在需要重跑时克隆；该目录已 gitignore
    python3 _vendor/hstu_probe.py
需要 torch。没有 torch 时以退出码 2 结束并给出安装命令。
克隆不在时同样以退出码 2 结束并给出克隆命令。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "generative-recommenders"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO.parent.parent / "执行时间预测"))

if not (REPO / "generative_recommenders").is_dir():
    print("没有官方代码。先克隆（该目录已 gitignore，不进仓库）：")
    print("  git clone --depth 1 https://github.com/meta-recsys/generative-recommenders \\")
    print("      _vendor/generative-recommenders")
    raise SystemExit(2)

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    print("没有 torch，先装：python3 -m pip install --user torch==2.2.2")
    raise SystemExit(2)


# ---------------------------------------------------------------- fbgemm 垫片
def _jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
    # 官方调用传的是 offsets=[x_offsets]（列表包一层），也有传裸张量的地方。
    if isinstance(offsets, (list, tuple)):
        offsets = offsets[0]
    batch = len(offsets) - 1
    width = values.size(-1)
    out = torch.full((batch, int(max_lengths[0]), width), padding_value, dtype=values.dtype)
    for row in range(batch):
        start, end = int(offsets[row]), int(offsets[row + 1])
        out[row, : end - start] = values[start:end]
    return out


def _dense_to_jagged(dense, offsets):
    if isinstance(offsets, (list, tuple)):
        offsets = offsets[0]
    pieces = []
    for row in range(len(offsets) - 1):
        start, end = int(offsets[row]), int(offsets[row + 1])
        pieces.append(dense[row, : end - start])
    return (torch.cat(pieces, dim=0),)


class _FakeFbgemm:
    jagged_to_padded_dense = staticmethod(_jagged_to_padded_dense)
    dense_to_jagged = staticmethod(_dense_to_jagged)


# ---------------------------------------------------------------- 记录器
EVENTS: list[tuple[str, tuple[tuple[int, ...], ...]]] = []


class _Proxy:
    """把 torch 转给真 torch，只替换 ops 并记录少数几个算子。"""

    def __init__(self, real):
        self._real = real
        self.ops = types.SimpleNamespace(fbgemm=_FakeFbgemm)
        self.nn = real.nn

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mm(self, a, b):
        EVENTS.append(("mm", (tuple(a.shape), tuple(b.shape))))
        return self._real.mm(a, b)

    def einsum(self, equation, *operands):
        EVENTS.append(("einsum", (equation, tuple(tuple(op.shape) for op in operands))))
        return self._real.einsum(equation, *operands)

    def bmm(self, a, b):
        EVENTS.append(("bmm", (tuple(a.shape), tuple(b.shape))))
        return self._real.bmm(a, b)


class _FProxy:
    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def layer_norm(self, x, normalized_shape, eps=1e-5):
        EVENTS.append(("layer_norm", (tuple(x.shape),)))
        return self._real.layer_norm(x, normalized_shape, eps=eps)

    def silu(self, x):
        EVENTS.append(("silu", (tuple(x.shape),)))
        return self._real.silu(x)


def _product(shape) -> int:
    count = 1
    for value in shape:
        count *= value
    return count


def main() -> int:
    from generative_recommenders.research.modeling.sequential import hstu as official

    official.torch = _Proxy(torch)
    official.F = _FProxy(F)

    from timepred.stage import StageLoad, StageSpec, features

    embedding_dim, linear_dim, attention_dim = 64, 32, 16
    heads, batch, length = 4, 2, 512

    class _ZeroBias(official.RelativeAttentionBiasModule):
        def forward(self, timestamps):
            return torch.zeros(batch, length, length)

    layer = official.SequentialTransductionUnitJagged(
        embedding_dim=embedding_dim,
        linear_hidden_dim=linear_dim,
        attention_dim=attention_dim,
        dropout_ratio=0.0,
        attn_dropout_ratio=0.0,
        num_heads=heads,
        linear_activation="silu",
        relative_attention_bias_module=_ZeroBias(),
        normalization="rel_bias",
    )
    layer.eval()

    tokens = batch * length
    x = torch.randn(tokens, embedding_dim)
    offsets = torch.arange(0, tokens + 1, length, dtype=torch.int32)
    # 官方在注意力里对 mask 做两次 unsqueeze，所以传 (N, N)。
    mask = torch.ones(length, length)

    with torch.no_grad():
        layer(x, offsets, None, mask)

    print(
        f"官方 HSTU 单层跑通：B={batch} L={length} H={heads} D={embedding_dim} "
        f"linear_dim={linear_dim} attention_dim={attention_dim}"
    )
    print()

    uvqk_width = linear_dim * 2 * heads + attention_dim * heads * 2
    records: list[tuple[str, str, int, str]] = []
    for name, shapes in EVENTS:
        if name == "mm":
            (m, k), (k2, n) = shapes
            records.append(("linear（uvqk 或 o 投影）", f"{m}x{k} @ {k2}x{n}", m * k * n, "乘加"))
        elif name == "einsum":
            equation, operands = shapes
            sizes: dict[str, int] = {}
            for letters, shape in zip(equation.split("->")[0].split(","), operands):
                for letter, size in zip(letters, shape):
                    sizes[letter] = size
            macs = 1
            for size in sizes.values():
                macs *= size
            out_elems = 1
            for letter in equation.split("->")[1]:
                out_elems *= sizes[letter]
            label = "self_attn（qkᵀ）" if "bhnm" in equation.split("->")[1] else "self_attn（p·v）"
            records.append((label, equation, macs, "乘加"))
            records.append((f"{label} 的输出", equation, out_elems, "元素"))
        elif name == "layer_norm":
            (shape,) = shapes
            records.append(("elemwise（layer_norm）", str(shape), _product(shape), "元素"))
        elif name == "silu":
            (shape,) = shapes
            records.append(("elemwise（silu）", str(shape), _product(shape), "元素"))

    print(f"{'记录到的算子':26s} {'形状 / 公式':44s} {'次数':>14s}  单位")
    for label, shape, value, unit in records:
        print(f"{label:26s} {shape:44s} {value:14,.0f}  {unit}")

    linear_values = [value for label, _, value, unit in records if label.startswith("linear")]
    qk_values = [value for label, _, value, _ in records if label == "self_attn（qkᵀ）"]
    pv_values = [value for label, _, value, _ in records if label == "self_attn（p·v）"]
    silu_values = [value for label, _, value, unit in records if "silu" in label]

    load = StageLoad(
        batch=batch,
        length=length,
        n_real=tokens,
        width=embedding_dim,
        hidden=uvqk_width,
        n_head=heads,
        d_head=attention_dim,
    )
    f_linear = features(StageSpec("proj", "linear"), load)["linear"]
    f_attn = features(StageSpec("attn", "self_attn"), load)["self_attn"]
    f_attn_v = features(
        StageSpec("attn", "self_attn", value_dim=linear_dim), load
    )["self_attn"]
    f_elem_sq = features(StageSpec("norm", "elemwise"), load)["elemwise_square"]

    print()
    print("框架的特征 vs 官方形状")
    print(f"  linear 特征（N_tok·d·hidden，hidden={uvqk_width}）    {f_linear:14,.0f}")
    print(f"  官方两次 mm 的乘加数之和                         {sum(linear_values):14,.0f}")
    print(f"  self_attn 特征（B·h·L²·d_head）                   {f_attn:14,.0f}")
    print(f"  官方 qkᵀ 的乘加数                                 {qk_values[0]:14,.0f}")
    print(f"  官方 p·v 的乘加数（V 维={linear_dim}）              {pv_values[0]:14,.0f}")
    print(f"  self_attn + value_dim={linear_dim}                {f_attn_v:14,.0f}")
    print(f"  qkᵀ + p·v                                        {qk_values[0] + pv_values[0]:14,.0f}")
    print(f"  elemwise_square 特征（2·B·h·L²）                  {f_elem_sq:14,.0f}")
    print(f"  官方 Silu 作用在 S×S 上的元素数                   {max(silu_values):14,.0f}")
    print()
    print("结论")
    print("  · 官方把四个投影合成一次 mm，故框架应对应两段 linear：uvqk 与 o；")
    print(f"    若合成一段，hidden 必须取 {uvqk_width} + {linear_dim * heads}，不能取 {embedding_dim}。")
    print(f"  · self_attn 特征与 qkᵀ 严格相等：{f_attn == qk_values[0]}")
    print(f"  · 但 p·v 的收缩维是 V（{linear_dim}），与 d_head（{attention_dim}）不同：")
    print(f"    qkᵀ + p·v = {qk_values[0] + pv_values[0]:,}，仅在 value_dim={linear_dim} 时对上。")
    print(f"    所以 L² 项的宽度是 d_head + value_dim；d_v = d_head 时退化为现式。")
    print(f"  · 官方对 S×S 分数做了一次 Silu，元素数 {max(silu_values):,}；")
    print(f"    elemwise_square/2 = {f_elem_sq // 2:,}，即该项与 qkᵀ 同阶，不能只按 B·L·d 记。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
