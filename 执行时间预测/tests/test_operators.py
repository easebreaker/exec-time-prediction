"""补登记的四类算子与五个登记位。

覆盖：
- elemwise：linear 与 square 两项；fused 时整段为 0
- moe_gate：renorm 改变打分列
- moe_comm：ep 切分的通信体积与跨度两档 v
- score_gemm：主项随商品数，不随 L
- 网络级截距：kernel 数 × 单 kernel 下发
- tile 模型：主项为块数，且能拟合回块单价
"""

from __future__ import annotations

import math
import unittest

from timepred.experiment import (
    ComplexityReport,
    NetworkFacts,
    StageObservation,
    complexity_grid,
    fit_complexity,
    fit_split,
    kernel_count,
    predict_network,
    stage_time,
)
from timepred.recipes import hstu_stages, hstu_stages_with_elemwise, moe_ffn_stages
from timepred.schema import FrameworkError
from timepred.stage import (
    StageLoad,
    StageSpec,
    StageSplit,
    default_load,
    features,
    volume_bytes,
)


def observe(spec, coefficients, loads, split, **kwargs):
    rows = []
    for load in loads:
        total, _, comm = stage_time(spec, coefficients, load, split, **kwargs)
        byte_count = None if split.kind == "none" else volume_bytes(spec, load, split)
        rows.append(StageObservation(load, split, total, byte_count, None if split.kind == "none" else comm))
    return rows


class FusedTests(unittest.TestCase):
    def test_fused_stage_is_all_zero(self) -> None:
        spec = StageSpec("norm", "elemwise", fused=True)
        self.assertTrue(all(value == 0.0 for value in features(spec, default_load()).values()))
        total, _, _ = stage_time(spec, {}, default_load(), StageSplit("none"))
        self.assertEqual(total, 0.0)

    def test_unfused_elemwise_has_a_square_term(self) -> None:
        spec = StageSpec("norm", "elemwise")
        load = StageLoad(batch=2, length=256, n_head=4, d_head=16)
        values = features(spec, load)
        self.assertEqual(values["elemwise_linear"], 2 * 2 * 256 * 64)
        self.assertEqual(values["elemwise_square"], 2 * 2 * 4 * 256 * 256)

    def test_fused_only_allowed_on_absorbable_operators(self) -> None:
        with self.assertRaises(FrameworkError):
            StageSpec("attn", "self_attn", fused=True)


class MoeGateTests(unittest.TestCase):
    def test_renorm_changes_only_the_score_column(self) -> None:
        load = StageLoad(batch=1, length=128, width=64, n_exp=64, topk=6)
        plain = features(StageSpec("gate", "moe_gate"), load)
        renorm = features(StageSpec("gate", "moe_gate", renorm=1), load)
        self.assertEqual(plain["gate_score"], 1 * 128 * 64 * 64)
        self.assertEqual(renorm["gate_score"], 1 * 128 * 6 * 64)
        self.assertEqual(plain["gate_sort"], renorm["gate_sort"])

    def test_gate_is_fitted_from_the_expert_axis(self) -> None:
        spec = StageSpec("gate", "moe_gate")
        truth = {"gate_score": 3e-9, "gate_sort": 2e-6, "intercept": 4e-3}
        report = fit_complexity(spec, observe(spec, truth, complexity_grid(spec), StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertLess(report.mape, 1e-6)
        self.assertNotIn("self_attn", report.coefficients())


class MoeCommTests(unittest.TestCase):
    def test_ep_volume_is_per_rank_and_scales_with_topk(self) -> None:
        spec = StageSpec("comm", "moe_comm")
        load = StageLoad(batch=2, length=256, width=64, topk=6)
        volume = volume_bytes(spec, load, StageSplit("ep", 8))
        expected = (8 - 1) / 8 * 1 * 2 * 256 * 6 * 64 * 2
        self.assertEqual(volume, expected)
        doubled = load.varied(topk=12)
        self.assertEqual(
            volume_bytes(spec, doubled, StageSplit("ep", 8)),
            2 * volume,
        )

    def test_two_spans_give_two_v(self) -> None:
        spec = StageSpec("comm", "moe_comm")
        truth = {"moe_route": 2e-6, "intercept": 0.0}
        rows = []
        for span, v in (("intra", 6e-9), ("cross", 5e-8)):
            for degree in (2, 4, 8):
                split = StageSplit("ep", degree, span=span)
                for length in (256, 1024):
                    load = StageLoad(batch=2, length=length)
                    total, _, comm = stage_time(spec, truth, load, split, tau_s=5e-5, v_s_per_byte=v)
                    rows.append(
                        StageObservation(load, split, total, volume_bytes(spec, load, split), comm)
                    )
        complexity = fit_complexity(
            spec, observe(spec, truth, complexity_grid(spec), StageSplit("none"))
        )
        fit = fit_split(spec, complexity, rows)
        self.assertTrue(fit.identified)
        self.assertAlmostEqual(fit.v_s_per_byte / 6e-9, 1.0, places=6)
        self.assertIsNotNone(fit.v_cross_s_per_byte)
        self.assertAlmostEqual(fit.v_cross_s_per_byte / 5e-8, 1.0, places=6)
        self.assertAlmostEqual(fit.tau_s / 5e-5, 1.0, places=6)

    def test_single_span_leaves_cross_empty(self) -> None:
        spec = StageSpec("comm", "moe_comm")
        truth = {"moe_route": 2e-6, "intercept": 0.0}
        rows = []
        for degree in (2, 4, 8):
            for length in (256, 1024):
                split = StageSplit("ep", degree)
                load = StageLoad(batch=2, length=length)
                total, _, comm = stage_time(spec, truth, load, split, tau_s=5e-5, v_s_per_byte=6e-9)
                rows.append(
                    StageObservation(load, split, total, volume_bytes(spec, load, split), comm)
                )
        complexity = fit_complexity(
            spec, observe(spec, truth, complexity_grid(spec), StageSplit("none"))
        )
        fit = fit_split(spec, complexity, rows)
        self.assertIsNone(fit.v_cross_s_per_byte)


class ScoreGemmTests(unittest.TestCase):
    def test_score_grows_with_items_not_with_length(self) -> None:
        spec = StageSpec("score", "score_gemm")
        base = StageLoad(batch=1, length=64, width=64, n_item=1000)
        longer = base.varied(length=4096)
        self.assertEqual(features(spec, base)["score"], features(spec, longer)["score"])
        self.assertEqual(features(spec, base.varied(n_item=2000))["score"], 2 * features(spec, base)["score"])

    def test_sampled_recall_replaces_item_count(self) -> None:
        spec = StageSpec("score", "score_gemm")
        load = StageLoad(batch=1, width=64, n_item=1_000_000, n_probe=4096)
        self.assertEqual(features(spec, load)["score"], 1 * 4096 * 64)


class NetworkInterceptTests(unittest.TestCase):
    def test_kernel_count_skips_fused_stages(self) -> None:
        self.assertEqual(kernel_count(hstu_stages()), 3)
        self.assertEqual(kernel_count(hstu_stages_with_elemwise()), 3)
        self.assertEqual(kernel_count(moe_ffn_stages()), 4)

    def test_network_intercept_is_added_once(self) -> None:
        stages = hstu_stages()
        coefficients = {"lookup": 1e-6, "linear": 3e-10, "self_attn": 6e-11}
        complexities = {}
        for spec in stages:
            rows = observe(spec, coefficients, complexity_grid(spec), StageSplit("none"))
            complexities[spec.name] = fit_complexity(spec, rows)
        facts = NetworkFacts(fixed_s=1e-4, dispatch_s=8e-6, sync_s=2e-5)
        with_b = predict_network(stages, complexities, {}, default_load(), network=facts)
        without = predict_network(stages, complexities, {}, default_load())
        self.assertEqual(with_b.n_kernel, 3)
        self.assertAlmostEqual(with_b.network_s, 1e-4 + 3 * 8e-6 + 2e-5)
        self.assertAlmostEqual(with_b.time_s - without.time_s, with_b.network_s)

    def test_network_intercept_mode_zeroes_the_stage_column(self) -> None:
        spec = StageSpec("attn", "self_attn", intercept_mode="network")
        self.assertEqual(features(spec, default_load())["intercept"], 0.0)


class TileModelTests(unittest.TestCase):
    def test_tile_primary_term_and_fit(self) -> None:
        spec = StageSpec("attn", "self_attn", model="tile", block_m=128, block_n=128)
        load = StageLoad(batch=1, length=1024, n_head=8, d_head=16)
        expected = 1 * 8 * math.ceil(1024 / 128) * math.ceil(1024 / 128)
        self.assertEqual(features(spec, load)["tile"], expected)
        self.assertEqual(features(spec, load)["self_attn"], 0.0)

        truth = {"tile": 7e-6, "intercept": 9e-3}
        rows = observe(spec, truth, complexity_grid(spec), StageSplit("none"))
        report = fit_complexity(spec, rows)
        self.assertTrue(report.matches_operator)
        self.assertIn("tile", report.coefficients())
        self.assertNotIn("self_attn", report.coefficients())
        self.assertLess(report.mape, 1e-6)

    def test_tile_model_requires_block_size(self) -> None:
        with self.assertRaises(FrameworkError):
            StageSpec("attn", "self_attn", model="tile")

    def test_tile_scales_with_the_split_degree(self) -> None:
        # cp_q 除 N、cp_qk 除 N²；未切分时不动。块数在大 L 上趋近 L²/N 与 L²/N²。
        spec = StageSpec("attn", "self_attn", model="tile", block_m=128, block_n=128)
        load = StageLoad(batch=1, length=4096, n_head=8, d_head=16)
        coefficients = {"tile": 7e-6, "intercept": 9e-3}
        plain, _, _ = stage_time(spec, coefficients, load, StageSplit("none"))
        shallow, _, _ = stage_time(spec, coefficients, load, StageSplit("cp_q", 4))
        deep, _, _ = stage_time(spec, coefficients, load, StageSplit("cp_qk", 4))
        block = features(spec, load)["tile"] * 7e-6
        self.assertAlmostEqual(plain - shallow, block - block / 4, places=9)
        self.assertAlmostEqual(plain - deep, block - block / 16, places=9)

    def test_sparse_multiplies_the_square_term(self) -> None:
        dense = StageSpec("attn", "self_attn")
        causal = StageSpec("attn", "self_attn", sparse=0.5)
        load = StageLoad(batch=1, length=512, n_head=4, d_head=16)
        self.assertEqual(
            features(causal, load)["self_attn"],
            0.5 * features(dense, load)["self_attn"],
        )


class ValueDimTests(unittest.TestCase):
    def test_value_dim_widens_the_square_term(self) -> None:
        load = StageLoad(batch=2, length=512, n_head=4, d_head=16)
        plain = features(StageSpec("attn", "self_attn"), load)["self_attn"]
        wide = features(StageSpec("attn", "self_attn", value_dim=32), load)["self_attn"]
        self.assertEqual(plain, 2 * 4 * 512 * 512 * 16)
        self.assertEqual(wide, 2 * 4 * 512 * 512 * (16 + 32))

    def test_value_dim_matches_the_official_hstu_shapes(self) -> None:
        # 官方 HSTU：qkᵀ 收缩 d_head=16，p·v 收缩 linear_dim=32。
        # 两个 einsum 的乘加数之和应等于 value_dim=32 时的 self_attn 特征。
        batch, length, heads, d_head, v_dim = 2, 512, 4, 16, 32
        macs = batch * heads * length * length * (d_head + v_dim)
        load = StageLoad(batch=batch, length=length, n_head=heads, d_head=d_head)
        spec = StageSpec("attn", "self_attn", value_dim=v_dim)
        self.assertEqual(features(spec, load)["self_attn"], macs)


class RecipeTests(unittest.TestCase):
    def test_decomposed_hstu_matches_original_when_fused(self) -> None:
        load = StageLoad(batch=1, length=1024, n_head=4, d_head=16, hidden=256, width=64)
        coefficients = {
            "lookup": 1e-6,
            "linear": 3e-10,
            "self_attn": 6e-11,
            "elemwise_linear": 4e-9,
            "elemwise_square": 2e-10,
        }
        plain = sum(
            stage_time(spec, coefficients, load, StageSplit("none"))[0] for spec in hstu_stages()
        )
        decomposed = sum(
            stage_time(spec, coefficients, load, StageSplit("none"))[0]
            for spec in hstu_stages_with_elemwise()
        )
        self.assertAlmostEqual(plain, decomposed, places=12)

    def test_moe_recipe_has_gate_and_two_comm_stages(self) -> None:
        operators = [spec.operator for spec in moe_ffn_stages()]
        self.assertEqual(operators.count("moe_comm"), 2)
        self.assertIn("moe_gate", operators)


if __name__ == "__main__":
    unittest.main()
