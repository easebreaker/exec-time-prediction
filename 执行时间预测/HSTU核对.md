# 官方 HSTU 单层的核对结果

> 对象：`meta-recsys/generative-recommenders` 的 `SequentialTransductionUnitJagged`
> （`generative_recommenders/research/modeling/sequential/hstu.py`，ICML'24 HSTU 的官方实现）。
> 本地浅克隆在 `_vendor/generative-recommenders/`；脚本为 `_vendor/hstu_probe.py`。
> 核对对象是**张量形状与乘加数**，不是硬件时间——本机无 GPU，谈时延没有意义。

---

## 一、做了什么

官方的这一层依赖 `torch.ops.fbgemm` 的 jagged 算子，而 `fbgemm_gpu` 没有 macOS 轮子。
脚本把两个 jagged 算子（`jagged_to_padded_dense`、`dense_to_jagged`）换成等价的 dense
实现，其余代码原样运行；再在 `mm`、`einsum`、`layer_norm`、`silu` 外面包一层记录形状。
因此核对的是**官方代码里真实发生的算子**，不是我们自己重写的复现版。

参数：`B=2, L=512, H=4, D=64, linear_dim=32, attention_dim=16`，一层。

## 二、官方一层展开后的算子与次数

| 算子 | 形状或公式 | 次数 | 单位 |
|---|---|---|---|
| `layer_norm` | `(1024, 64)` | 65,536 | 元素 |
| `mm`（uvqk 四投影合一） | `1024×64 @ 64×384` | 25,165,824 | 乘加 |
| `silu` | `(1024, 384)` | 393,216 | 元素 |
| `einsum`（qkᵀ） | `bnhd,bmhd->bhnm` | 33,554,432 | 乘加 |
| qkᵀ 输出 | `bnhd,bmhd->bhnm` | 2,097,152 | 元素 |
| `silu`（作用在 S×S 上） | `(2, 4, 512, 512)` | 2,097,152 | 元素 |
| `einsum`（p·v） | `bhnm,bmhd->bnhd` | 67,108,864 | 乘加 |
| p·v 输出 | `bhnm,bmhd->bnhd` | 131,072 | 元素 |
| `layer_norm`（o 之前） | `(1024, 128)` | 131,072 | 元素 |

`384 = 2·32·4 + 2·16·4`（u、v、q、k 四个投影合在一个权重里）；
`128 = 32·4`（o 投影的输入宽度）。

## 三、与框架特征的对照

| 框架特征 | 值 | 官方对应 | 是否相等 |
|---|---|---|---|
| `linear`（`N_tok·d·hidden`，`hidden=384`） | 25,165,824 | uvqk 的 `mm` | **相等** |
| `self_attn`（`B·h·L²·d_head`） | 33,554,432 | qkᵀ 的乘加数 | **相等** |
| `self_attn` + `value_dim=32` | 100,663,296 | qkᵀ + p·v | **相等** |
| `elemwise_square`（`2·B·h·L²`） | 4,194,304 | S×S 上 `silu` 的元素数 ×2 | **相等**（口径差 2 倍：读+写） |

## 四、三条结论（都已落进代码）

**1. `linear` 必须用「权重宽度」而不是隐藏维。**
官方把 u、v、q、k 四个投影合成一次 `mm`，输出宽度 `2·linear_dim·H + 2·attention_dim·H`。
用 `hidden=64`（`embedding_dim`）会低估 6 倍。另有一次 o 投影（输入宽度 `linear_dim·H`），
所以一层对应**两段 linear**。→ 框架不需要改，只需填表时把 `hidden` 填对。

**2. L² 项的宽度是 `d_head + V`，不是 `d_head`。**
qkᵀ 收缩 `d_head`，p·v 收缩 V 维。HSTU 里 `linear_dim(32) ≠ attention_dim(16)`，
两者之和才是这一层的全部 L² 乘加。→ 新增登记位 `value_dim`：
`self_attn` 特征改为 `B·h·L²·(d_head + value_dim)`，`value_dim=0` 时退化为原式。
`tests/test_operators.py` 里有对应用例。

**3. 对 S×S 做逐元素是真实的、且与 qkᵀ 同阶。**
官方在 `qk_attn * mask` 之前做了一次 `F.silu`，作用在 `(B, H, L, L)` 上。
`B·H·L² = 2,097,152`，而 qkᵀ 的乘加是 `B·H·L²·d_head = 33.5M`；即这一项是它的 `1/d_head`，
但**它是 L² 阶，不是 L 阶**。若按 `B·L·d` 记（`1024×64 = 65,536`），会低估 32 倍。
→ 新增 `elemwise_square` 列，`elemwise` 段可以同时挂 `elemwise_linear` 与 `elemwise_square`。

## 五、未覆盖的部分

| 项 | 原因 |
|---|---|
| 真实时延 | 本机无 GPU，且官方 kernel（`hstu_attention` Triton/CUDA）只在加速卡上编译 |
| 多层的 kernel 数 | 一层已能确定每层算子数；`kernels_per_layer` 需按实际融合情况填 |
| `fbgemm` jagged 的搬运开销 | 垫片是 dense 实现，字节数与真实 jagged 布局不同，故不用于通信量核对 |
| 反向 | 只跑了前向 |

## 六、复现

```bash
python3 -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.2.2
cd 执行时间预测 && python3 -m timepred.flow_demo      # 框架全流程
cd .. && OMP_NUM_THREADS=1 python3 _vendor/hstu_probe.py  # 官方 HSTU 核对
```

macOS 上若报 `OMP: Error #179 ... SHM2`，是沙箱对共享内存的限制，加 `OMP_NUM_THREADS=1`
或在沙箱外运行即可。
