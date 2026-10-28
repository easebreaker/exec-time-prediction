"""合成的一个模型、多种切分、多组负载。只用来核对公式和拟合，不是实测。"""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import BlockBaseline
from .predict import SchemeParams, TimeModel
from .schema import BlockSpec, Load, ModelSpec, Plan, Sample, Split


def make_spec(capacity_bytes: int = 16 * 1024**3) -> ModelSpec:
    return ModelSpec(
        blocks=(
            BlockSpec("embedding", "embedding", d_model=64, table_bytes=4 * 1024**3),
            BlockSpec("cross", "cross", d_model=64, param_bytes=8 * 1024**2),
            BlockSpec("attention", "attention", d_model=64, n_layers=4, param_bytes=128 * 1024**2),
            BlockSpec("ffn", "ffn", d_model=64, n_layers=4, param_bytes=256 * 1024**2),
        ),
        capacity_bytes=capacity_bytes,
        elem_bytes=2,
    )


def make_load(batch: int, length: int) -> Load:
    return Load(
        batch,
        {"attention": length, "ffn": length},
        {"embedding": 256, "cross": 64},
    )


def make_plans() -> dict[str, Plan]:
    names = ("embedding", "cross", "attention", "ffn")
    return {
        "none": Plan({}),
        "dp": Plan({name: Split("dp", 8) for name in names}),
        "cp2": Plan({"attention": Split("cp", 2)}),
        "cp4": Plan({"attention": Split("cp", 4)}),
        "cp8": Plan({"attention": Split("cp", 8)}),
        "embp2": Plan({"embedding": Split("embp", 2)}),
        "embp": Plan({"embedding": Split("embp", 4)}),
        "embp8": Plan({"embedding": Split("embp", 8)}),
        "tp": Plan({"attention": Split("tp", 4), "ffn": Split("tp", 4)}),
        "pp": Plan({"attention": Split("pp", 4, 8), "ffn": Split("pp", 4, 8)}),
        "mixed": Plan({"embedding": Split("embp", 4), "attention": Split("cp", 8)}),
    }


def truth_model(spec: ModelSpec | None = None) -> TimeModel:
    spec = spec or make_spec()
    baselines = {
        "embedding": BlockBaseline((2e-5, 0.02)),
        "cross": BlockBaseline((1e-5, 0.01)),
        "attention": BlockBaseline((2e-10, 5e-7, 1e-4)),
        "ffn": BlockBaseline((1e-7, 1e-3)),
    }
    schemes = {
        "cp": SchemeParams(kappa=0.4, tau_s=0.03, v_s_per_byte=1e-8),
        "embp": SchemeParams(kappa=0.2, tau_s=0.004, v_s_per_byte=1e-7),
        "tp": SchemeParams(kappa=0.15, tau_s=0.003, v_s_per_byte=1e-9),
        "pp": SchemeParams(kappa=0.0, tau_s=0.002, v_s_per_byte=1e-9),
    }
    return TimeModel(spec, baselines, schemes)


@dataclass(frozen=True)
class Problem:
    spec: ModelSpec
    truth: TimeModel
    loads: list[Load]
    plans: dict[str, Plan]
    samples: list[Sample]


def build_problem() -> Problem:
    spec = make_spec()
    truth = truth_model(spec)
    plans = make_plans()
    loads = [make_load(batch, length) for batch in (1, 10) for length in (512, 4096, 8192)]
    samples: list[Sample] = []
    for load in loads:
        for name, plan in plans.items():
            breakdown = truth.predict(load, plan)
            block_times = breakdown.block_t0_s if name in ("none", "dp") else None
            samples.append(
                Sample(
                    load,
                    plan,
                    breakdown.time_s,
                    block_times,
                    breakdown.mem_bytes,
                )
            )
    return Problem(spec, truth, loads, plans, samples)
