"""原理性测定的判定树：每条判据在「两种相反输入」上必须给出相反结论。"""

from __future__ import annotations

import math
import unittest

from timepred.principle import (
    Reading,
    by_experiment,
    experiment_plan,
    identify_e1a,
    identify_e2,
    identify_e3,
    identify_e4,
    identify_e5,
    infer_efficiency,
    verdicts_to_model,
)

BM = 128


def blocks(length: int, degree: int, block: int = BM) -> int:
    return math.ceil(length / degree / block) * math.ceil(length / block)


class PlanTests(unittest.TestCase):
    def test_plan_covers_six_experiments_and_costs_31(self) -> None:
        trials = experiment_plan()
        self.assertEqual(len(trials), 31)
        grouped = by_experiment(trials)
        self.assertEqual(
            sorted(grouped),
            ["E1a", "E1c", "E2", "E3a", "E3b", "E4", "E5"],
        )

    def test_e3a_uses_the_high_contrast_cell(self) -> None:
        # cp8 的首对跳幅 +14.3%，cp2 只有 +4.7%（不可测）。
        grouped = by_experiment(experiment_plan())
        degrees = {trial.split.degree for trial in grouped["E3a"]}
        self.assertEqual(degrees, {8})
        lengths = [trial.load.length for trial in grouped["E3a"]]
        steps = [blocks(length, 8) for length in lengths]
        self.assertGreater(steps[1] / steps[0] - 1.0, 0.10)
        # 跨格发生在 Q 侧：L/N 不是 128 的倍数
        self.assertTrue(any((length // 8) % BM for length in lengths))

    def test_plan_includes_n_equals_one(self) -> None:
        grouped = by_experiment(experiment_plan())
        degrees = {trial.split.degree for trial in grouped["E5"]}
        self.assertEqual(degrees, {1})


class E1Tests(unittest.TestCase):
    def test_per_layer_dispatch_is_detected(self) -> None:
        verdict = identify_e1a([(1, 2.2e-3), (2, 4.4e-3), (4, 8.8e-3)])
        self.assertIn("逐 kernel 下发", verdict.answer)
        self.assertEqual(verdict.confidence, "高")

    def test_constant_offset_is_reported(self) -> None:
        verdict = identify_e1a([(1, 3.0e-3), (2, 4.4e-3), (4, 7.2e-3)])
        self.assertIn("常数项", verdict.answer)

    def test_single_point_cannot_decide(self) -> None:
        self.assertEqual(identify_e1a([(1, 2.2e-3)]).confidence, "无法判定")


class E2Tests(unittest.TestCase):
    def test_both_gemms_equal_weight(self) -> None:
        verdict = identify_e2([(64, 64, 1.0), (128, 64, 1.5), (64, 128, 1.5)])
        self.assertIn("d_head + d_v", verdict.answer)

    def test_head_invariant_means_not_gemm(self) -> None:
        verdict = identify_e2([(64, 64, 1.0), (128, 64, 1.0), (64, 128, 1.0)])
        self.assertIn("主项不是 GEMM", verdict.answer)

    def test_value_dim_only_moves_value_side(self) -> None:
        # 只有 p·v 计费：d_v 加倍 → 时间上升，d_head 加倍 → 不变
        verdict = identify_e2([(64, 64, 1.0), (128, 64, 1.0), (64, 128, 1.5)])
        self.assertEqual(verdict.confidence, "中")


class E3Tests(unittest.TestCase):
    def _readings(self, steps: list[int], base: float, slope: float, sigma: float, stepped: bool):
        times = [
            base + slope * (s if stepped else steps[0] * (1.0315 ** i))
            for i, s in enumerate(steps)
        ]
        return [Reading(8192 + 128 * i, t, sigma * t) for i, t in enumerate(times)]

    def test_step_is_visible_with_low_noise(self) -> None:
        steps = [512, 585, 594]
        verdict = identify_e3(self._readings(steps, 0.3e-3, 7.5e-6, 0.02, True), steps)
        self.assertIn("台阶成立", verdict.answer)

    def test_smooth_is_rejected_with_low_noise(self) -> None:
        steps = [512, 585, 594]
        verdict = identify_e3(self._readings(steps, 0.3e-3, 7.5e-6, 0.02, False), steps)
        self.assertIn("台阶不成立", verdict.answer)

    def test_high_noise_reports_unmeasurable_not_absent(self) -> None:
        steps = [512, 585, 594]
        verdict = identify_e3(self._readings(steps, 0.3e-3, 7.5e-6, 0.115, True), steps)
        self.assertIn("不可测", verdict.answer)
        self.assertNotIn("不成立", verdict.answer)


class E4Tests(unittest.TestCase):
    def test_linear_scaling(self) -> None:
        readings = [Reading(x, 0.8e-3 * x, 0.02e-3 * x) for x in (1, 2, 4, 8, 16, 32)]
        self.assertIn("线性", identify_e4(readings).answer)

    def test_plateau_is_reported(self) -> None:
        readings = [Reading(x, 0.8e-3 * max(1.0, x / 8), 0.02e-3) for x in (1, 2, 4, 8, 16, 32)]
        self.assertIn("平坦", identify_e4(readings).answer)


class E5Tests(unittest.TestCase):
    def test_transition_is_located(self) -> None:
        ladder = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
        l_star = 1200.0
        readings = [
            Reading(
                L,
                0.5e-3 + 8e-3 * L / l_star if L < l_star else 0.5e-3 + 8e-3 * (L / l_star) ** 2,
                0.05e-3,
            )
            for L in ladder
        ]
        verdict = identify_e5(readings)
        self.assertIn("过渡发生在", verdict.answer)
        located = float(verdict.answer.split("L≈")[1].split("（")[0])
        self.assertTrue(l_star / 3 < located < l_star * 3)

    def test_pure_quadratic_has_no_transition(self) -> None:
        ladder = (256, 512, 1024, 2048, 4096, 8192, 16384)
        readings = [Reading(L, 1e-3 * (L / 256) ** 2, 0.01e-3) for L in ladder]
        self.assertIn("未见", identify_e5(readings).answer)

    def test_efficiency_inversion(self) -> None:
        eff = infer_efficiency(1200.0, 245e12, 1.6e12)
        self.assertAlmostEqual(eff["ideal_l_star"], 306.25, places=2)
        self.assertAlmostEqual(eff["ratio"], 1200.0 / 306.25, places=6)
        # 反解出的有效值应与实测 L* 自洽
        self.assertAlmostEqual(eff["p_eff"] / eff["bw_eff"] * 2.0, eff["ideal_l_star"], places=6)


class ModelInstructionTests(unittest.TestCase):
    def test_verdicts_map_to_concrete_instructions(self) -> None:
        verdicts = [
            identify_e1a([(1, 2.2e-3), (2, 4.4e-3), (4, 8.8e-3)]),
            identify_e2([(64, 64, 1.0), (128, 64, 1.5), (64, 128, 1.5)]),
            identify_e3(
                [Reading(8192 + 128 * i, 0.3e-3 + 7.5e-6 * s, 0.02 * (0.3e-3 + 7.5e-6 * s))
                 for i, s in enumerate((512, 585, 594))],
                [512, 585, 594],
            ),
            identify_e4([Reading(x, 0.8e-3 * x, 0.02e-3 * x) for x in (1, 2, 4, 8)]),
        ]
        out = verdicts_to_model(verdicts)
        self.assertIn("n_kernel", out["b"])
        self.assertIn("value_dim", out["l2_width"])
        self.assertIn("ceil", out["ceil"])
        self.assertIn("B·H", out["parallel"])


if __name__ == "__main__":
    unittest.main()
