"""用合成的阶段运行把复杂度、加速比和通信拟合走一遍。

这些时间由公式生成，用来核对实验流程，不是实测。
"""

from __future__ import annotations

from .experiment import (
    StageObservation,
    complexity_grid,
    fit_complexity,
    fit_split,
    predict_network,
    stage_time,
)
from .recipes import hstu_stages, token_mixer_stage
from .stage import StageSplit, default_load, volume_bytes


def _measure(spec, coefficients, loads, split, **kwargs) -> list[StageObservation]:
    observations = []
    for load in loads:
        total, _, comm = stage_time(spec, coefficients, load, split, **kwargs)
        if split.kind == "none":
            observations.append(StageObservation(load, split, total))
        else:
            observations.append(
                StageObservation(load, split, total, volume_bytes(spec, load, split), comm)
            )
    return observations


def _show_complexity(title: str, spec, coefficients: dict[str, float]) -> None:
    observations = _measure(spec, coefficients, complexity_grid(spec), StageSplit("none"))
    report = fit_complexity(spec, observations)
    terms = "、".join(f"{name}={value:.3g}" for name, value in report.terms) or "无"
    length = "随 L 变化" if report.depends_on_length else "不随 L 变化"
    print(f"{title}: {terms}，{length}，与标称算子一致={report.matches_operator}")


def main() -> None:
    print("阶段实验用的是合成时间，只核对流程。")
    _show_complexity(
        "自注意力",
        hstu_stages()[2],
        {"self_attn": 2e-9, "intercept": 0.02},
    )
    mixer = token_mixer_stage(False)
    _show_complexity("token 轴矩阵乘", mixer, {"token_square": 1e-6})
    _show_complexity(
        "组数随 L/T 变化的组内注意力",
        token_mixer_stage(True),
        {"token_square": 1e-6},
    )

    attn = hstu_stages()[2]
    coefficients = {"self_attn": 2e-9, "intercept": 0.02}
    plain = _measure(attn, coefficients, complexity_grid(attn), StageSplit("none"))
    complexity = fit_complexity(attn, plain)
    split_loads = [default_load().varied(length=length) for length in (512, 2048)]
    split_obs = []
    for degree in (2, 4, 8):
        split_obs.extend(
            _measure(
                attn,
                coefficients,
                split_loads,
                StageSplit("cp_q", degree),
                kappa=0.25,
                tau_s=0.01,
                v_s_per_byte=1e-6,
            )
        )
    fitted = fit_split(attn, complexity, split_obs)
    print(
        f"CP 只切 Query: κ={fitted.kappa:.3g}，τ={fitted.tau_s:.3g}，"
        f"v={fitted.v_s_per_byte:.3g}，分开={fitted.identified}，重叠={fitted.overlap}"
    )
    one_length = [obs for obs in split_obs if obs.load.length == 512]
    collapsed = fit_split(attn, complexity, one_length)
    print(f"只有一个长度时: {collapsed.reason or '已分开'}")

    stages = hstu_stages()
    load = default_load()
    network = predict_network(
        stages,
        {
            "embed": fit_complexity(
                stages[0],
                _measure(stages[0], {"lookup": 1e-5, "intercept": 0.01}, complexity_grid(stages[0]), StageSplit("none")),
            ),
            "proj": fit_complexity(
                stages[1],
                _measure(stages[1], {"linear": 1e-8}, complexity_grid(stages[1]), StageSplit("none")),
            ),
            "attn": complexity,
        },
        {"attn": fitted},
        load,
        {"attn": StageSplit("cp_q", 4)},
    )
    print(
        "HSTU 三段相加，只切注意力: "
        + "，".join(
            f"{name}={piece.time_s:.4f}s（{piece.split.kind}）" for name, piece in network.stages
        )
        + f"，合计={network.time_s:.4f}s"
    )


if __name__ == "__main__":
    main()
