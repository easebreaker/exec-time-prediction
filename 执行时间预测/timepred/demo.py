"""跑通合成样本上的留出、拟合和测试。"""

from __future__ import annotations

from .evaluate import evaluate
from .fit import fit
from .holdout import holdout_by_load
from .synthetic import build_problem, make_load


def main() -> None:
    problem = build_problem()
    held = holdout_by_load(problem.samples, test_fraction=0.2, seed=0)
    report = fit(problem.spec, held.train)
    checked = evaluate(report.model, held.test)
    print("合成数据，只检查公式和拟合流程，不是实测。")
    print(
        f"训练集 {len(held.train)} 条，测试集 {len(held.test)} 条，"
        f"留出 {len(held.test_keys)} 个负载组合。"
    )
    print(
        f"对照 {report.n_control} 条，混合切分未用于拟合 {report.n_mixed_unused} 条。"
    )
    for kind in ("cp", "embp", "tp", "pp"):
        truth = problem.truth.schemes[kind]
        got = report.model.schemes[kind]
        inactive = "、".join(report.inactive_parameters.get(kind, ())) or "无"
        print(
            f"{kind}: κ {got.kappa:.6g}（真值 {truth.kappa:.6g}），"
            f"τ {got.tau_s:.6g}（真值 {truth.tau_s:.6g}），"
            f"v {got.v_s_per_byte:.6g}（真值 {truth.v_s_per_byte:.6g}），"
            f"未估参数 {inactive}"
        )
    print(
        f"测试集平均绝对对数误差 {checked.mean_abs_log_error:.3e}，"
        f"MAPE {checked.mape:.3e}，"
        f"方案排序命中 {checked.rank_accuracy}"
    )
    for length in (512, 8192):
        load = make_load(1, length)
        breakdown = report.model.predict(load, problem.plans["cp8"])
        print(
            f"CP=8, batch=1, L={length}: "
            f"r={breakdown.r:.3f}, g={breakdown.g:.3f}, c={breakdown.c:.3f}, "
            f"预测时间={breakdown.time_s:.4f}s, 加速比={breakdown.speedup:.3f}, "
            f"上界={breakdown.speedup_upper_bound:.3f}"
        )


if __name__ == "__main__":
    main()
