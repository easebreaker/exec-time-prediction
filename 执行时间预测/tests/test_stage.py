"""阶段实验：单项识别、切分是否能把 κ 和 v 分开、阶段时间相加。"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from timepred.experiment import (
    SplitReport,
    StageObservation,
    complexity_grid,
    fit_complexity,
    fit_split,
    predict_network,
    predict_stage,
    stage_time,
)
from timepred.linalg import column_rank
from timepred.recipes import hyformer_stages, hstu_stages, longer_stages, rankmixer_stages
from timepred.schema import FrameworkError
from timepred.stage import (
    StageLoad,
    StageSpec,
    StageSplit,
    default_load,
    features,
    volume_bytes,
)
from timepred.stage_io import load_stage_samples, write_stage_template


def assert_rel(test: unittest.TestCase, got: float, expected: float) -> None:
    scale = max(abs(expected), 1e-30)
    test.assertLess(abs(got - expected), 1e-3 * scale + 1e-8, msg=f"{got} != {expected}")


def measure(spec, coefficients, loads, split, **kwargs) -> list[StageObservation]:
    observations = []
    for load in loads:
        total, _, comm = stage_time(spec, coefficients, load, split, **kwargs)
        bytes_ = None if split.kind == "none" else volume_bytes(spec, load, split)
        comm_time = None if split.kind == "none" else comm
        observations.append(StageObservation(load, split, total, bytes_, comm_time))
    return observations


class FeatureTests(unittest.TestCase):
    def test_self_attn_uses_padded_length_and_linear_uses_real_tokens(self) -> None:
        attn = StageSpec("attn", "self_attn")
        proj = StageSpec("proj", "linear")
        dense = StageLoad(batch=2, length=100, n_real=200, n_head=2, d_head=4, width=8, hidden=8)
        jagged = dense.varied(n_real=50)
        self.assertEqual(features(attn, dense)["self_attn"], features(attn, jagged)["self_attn"])
        self.assertEqual(features(attn, dense)["self_attn"], 2 * 2 * 100 * 100 * 4)
        self.assertEqual(features(attn, dense)["attn_linear"], 2 * 2 * 100 * 4)
        self.assertEqual(features(proj, dense)["linear"], 200 * 8 * 8)
        self.assertEqual(features(proj, jagged)["linear"], 50 * 8 * 8)

    def test_context_parallel_volume_and_group_split_has_none(self) -> None:
        attn = StageSpec("attn", "self_attn", elem_bytes=2)
        load = StageLoad(batch=1, length=4, width=8)
        self.assertEqual(volume_bytes(attn, load, StageSplit("cp_q", 2)), 0.5 * 1 * 4 * 8 * 2)
        mixer = StageSpec("mix", "token_gemm")
        self.assertEqual(volume_bytes(mixer, load, StageSplit("shard_r", 4)), 0.0)

    def test_query_split_rejected_on_a_projection(self) -> None:
        proj = StageSpec("proj", "linear")
        with self.assertRaises(FrameworkError):
            volume_bytes(proj, default_load(), StageSplit("cp_q", 2))

    def test_column_rank_collapses_proportional_columns(self) -> None:
        self.assertEqual(column_rank([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), 1)
        self.assertEqual(column_rank([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 2)


class ComplexityTests(unittest.TestCase):
    def test_self_attention_keeps_the_square_term(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9, "intercept": 0.02}
        report = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertTrue(report.depends_on_length)
        self.assertIn("self_attn", report.coefficients())
        self.assertNotIn("attn_linear", report.coefficients())
        self.assertNotIn("token_square", report.coefficients())
        assert_rel(self, report.coefficients()["self_attn"], 2e-9)
        self.assertLess(report.mape, 1e-4)

    def test_sdpa_keeps_the_linear_length_term(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficient = 2e-6
        coefficients = {"attn_linear": coefficient, "intercept": 0.01}
        report = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertTrue(report.depends_on_length)
        self.assertIn("attn_linear", report.coefficients())
        self.assertNotIn("self_attn", report.coefficients())
        self.assertNotIn("linear", report.coefficients())
        assert_rel(self, report.coefficients()["attn_linear"], coefficient)
        longer = default_load().varied(length=8192, n_head=8)
        truth, _, _ = stage_time(spec, coefficients, longer, StageSplit("none"))
        predicted, _, _ = stage_time(spec, report.coefficients(), longer, StageSplit("none"))
        assert_rel(self, predicted, truth)
        length_only = [default_load().varied(length=length) for length in (128, 256, 512, 1024, 2048, 4096)]
        length_report = fit_complexity(spec, measure(spec, {"attn_linear": coefficient}, length_only, StageSplit("none")))
        self.assertEqual(set(length_report.coefficients()), {"attn_linear"})

    def test_sdpa_between_linear_and_square_keeps_both_terms(self) -> None:
        spec = StageSpec("attn", "self_attn")
        # L=4096 处对数斜率约 1.3，更短的序列更接近 1。
        square = 2e-9
        linear = (7.0 / 3.0) * square * 4096
        coefficients = {"self_attn": square, "attn_linear": linear}
        report = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertIn("self_attn", report.coefficients())
        self.assertIn("attn_linear", report.coefficients())
        assert_rel(self, report.coefficients()["self_attn"], square)
        assert_rel(self, report.coefficients()["attn_linear"], linear)
        short = default_load().varied(length=256)
        long = default_load().varied(length=4096)
        short_time, _, _ = stage_time(spec, coefficients, short, StageSplit("none"))
        long_time, _, _ = stage_time(spec, coefficients, long, StageSplit("none"))
        slope = math.log(long_time / short_time) / math.log(4096 / 256)
        self.assertGreater(slope, 1.0)
        self.assertLess(slope, 1.35)
        predicted, _, _ = stage_time(spec, report.coefficients(), long, StageSplit("none"))
        assert_rel(self, predicted, long_time)

    def test_token_gemm_does_not_depend_on_sequence_length(self) -> None:
        spec = StageSpec("mixer", "token_gemm")
        report = fit_complexity(
            spec,
            measure(spec, {"token_square": 1e-6}, complexity_grid(spec), StageSplit("none")),
        )
        self.assertTrue(report.matches_operator)
        self.assertTrue(report.length_checked)
        self.assertFalse(report.depends_on_length)
        self.assertEqual(set(report.coefficients()), {"token_square"})

    def test_groups_tied_to_length_make_the_square_term_track_length(self) -> None:
        spec = StageSpec("inner", "group_attn", groups_follow_length=True)
        report = fit_complexity(
            spec,
            measure(spec, {"token_square": 1e-6}, complexity_grid(spec), StageSplit("none")),
        )
        self.assertTrue(report.depends_on_length)
        self.assertIn("token_square", report.coefficients())
        self.assertNotIn("self_attn", report.coefficients())

    def test_permute_scales_with_width_and_projection_scales_with_width_squared(self) -> None:
        permute = StageSpec("mix", "token_permute")
        linear = StageSpec("ffn", "linear", token_axis="fixed")
        permute_report = fit_complexity(
            permute,
            measure(permute, {"permute": 1e-6}, complexity_grid(permute), StageSplit("none")),
        )
        linear_report = fit_complexity(
            linear,
            measure(linear, {"linear": 1e-8}, complexity_grid(linear), StageSplit("none")),
        )
        self.assertAlmostEqual(permute_report.width_log_slope or 0.0, 1.0, places=2)
        self.assertAlmostEqual(linear_report.width_log_slope or 0.0, 2.0, places=2)
        self.assertFalse(permute_report.depends_on_length)

    def test_batch_only_sweep_keeps_the_length_term(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9, "intercept": 0.02}
        loads = [default_load().varied(batch=batch) for batch in (1, 2, 4, 8)]
        report = fit_complexity(spec, measure(spec, coefficients, loads, StageSplit("none")))
        self.assertIn("self_attn", report.coefficients())
        self.assertNotIn("lookup", report.coefficients())
        self.assertNotIn("cross_attn", report.coefficients())
        long = default_load().varied(length=8192)
        truth, _, _ = stage_time(spec, coefficients, long, StageSplit("none"))
        predicted, _, _ = stage_time(spec, report.coefficients(), long, StageSplit("none"))
        assert_rel(self, predicted, truth)

        cross = StageSpec("decode", "cross_attn")
        cross_coef = {"cross_attn": 1e-8, "intercept": 0.01}
        cross_report = fit_complexity(cross, measure(cross, cross_coef, loads, StageSplit("none")))
        self.assertIn("cross_attn", cross_report.coefficients())
        self.assertNotIn("lookup", cross_report.coefficients())
        longer_kv = default_load().varied(length_kv=8192)
        truth, _, _ = stage_time(cross, cross_coef, longer_kv, StageSplit("none"))
        predicted, _, _ = stage_time(cross, cross_report.coefficients(), longer_kv, StageSplit("none"))
        assert_rel(self, predicted, truth)

    def test_proportional_ids_do_not_replace_permute(self) -> None:
        spec = StageSpec("boost_mix", "token_permute", axes={"tokens": "boost_tokens"})
        base = default_load().varied(extras={"boost_tokens": 16})
        loads = []
        for batch in (1, 2, 4, 8):
            for tokens in (8, 16, 32):
                loads.append(
                    base.varied(
                        batch=batch,
                        n_id=tokens * base.width,
                        extras={"boost_tokens": tokens},
                    )
                )
        report = fit_complexity(spec, measure(spec, {"permute": 1e-6}, loads, StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertNotIn("lookup", report.coefficients())
        shifted = base.varied(n_id=128, extras={"boost_tokens": 64})
        truth, _, _ = stage_time(spec, {"permute": 1e-6}, shifted, StageSplit("none"))
        predicted, _, _ = stage_time(spec, report.coefficients(), shifted, StageSplit("none"))
        assert_rel(self, predicted, truth)

    def test_bound_token_axis_is_swept_for_boost(self) -> None:
        spec = StageSpec("boost_mix", "token_permute", axes={"tokens": "boost_tokens"})
        loads = complexity_grid(spec)
        values = {load.extras["boost_tokens"] for load in loads}
        self.assertGreater(len(values), 1)
        report = fit_complexity(spec, measure(spec, {"permute": 1e-6}, loads, StageSplit("none")))
        self.assertTrue(report.matches_operator)
        self.assertNotIn("lookup", report.coefficients())

    def test_tiny_scaled_coefficient_is_kept_for_the_primary_term(self) -> None:
        spec = StageSpec("attn", "self_attn")
        loads = complexity_grid(spec)
        peak = max(features(spec, load)["self_attn"] for load in loads)
        intercept = 0.05
        coefficient = 0.2 * 1e-5 * intercept / peak
        coefficients = {"self_attn": coefficient, "intercept": intercept}
        report = fit_complexity(spec, measure(spec, coefficients, loads, StageSplit("none")))
        self.assertIn("self_attn", report.coefficients())
        heavy = default_load().varied(batch=128, length=65536, n_head=64, d_head=128, width=512, hidden=2048)
        truth, _, _ = stage_time(spec, coefficients, heavy, StageSplit("none"))
        predicted, _, _ = stage_time(spec, report.coefficients(), heavy, StageSplit("none"))
        self.assertGreater(truth / intercept, 4.0)
        assert_rel(self, predicted, truth)

    def test_one_short_sample_does_not_flatten_the_stage(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9, "intercept": 0.02}
        loads = complexity_grid(spec)
        observations = measure(spec, coefficients, loads, StageSplit("none"))
        peak_index = max(
            range(len(observations)),
            key=lambda index: features(spec, observations[index].load)["self_attn"],
        )
        median = sorted(obs.time_s for obs in observations)[len(observations) // 2]
        observations[peak_index] = StageObservation(
            observations[peak_index].load,
            StageSplit("none"),
            median * 1e-4,
        )
        report = fit_complexity(spec, observations)
        self.assertIn("self_attn", report.coefficients())
        truth, _, _ = stage_time(spec, coefficients, default_load(), StageSplit("none"))
        predicted, _, _ = stage_time(spec, report.coefficients(), default_load(), StageSplit("none"))
        self.assertLess(abs(predicted - truth) / truth, 0.05)


class SplitTests(unittest.TestCase):
    def test_query_split_recovers_kappa_tau_and_bandwidth(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9, "intercept": 0.02}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        loads = [default_load().varied(length=length) for length in (512, 2048)]
        observations = []
        for degree in (2, 4, 8):
            observations.extend(
                measure(
                    spec,
                    coefficients,
                    loads,
                    StageSplit("cp_q", degree),
                    kappa=0.25,
                    tau_s=0.01,
                    v_s_per_byte=1e-6,
                )
            )
        report = fit_split(spec, complexity, observations)
        self.assertTrue(report.identified)
        self.assertFalse(report.overlap)
        assert_rel(self, report.kappa, 0.25)
        assert_rel(self, report.tau_s, 0.01)
        assert_rel(self, report.v_s_per_byte, 1e-6)
        self.assertIsNone(report.degree)
        one_length = [obs for obs in observations if obs.load.length == 512]
        collapsed = fit_split(spec, complexity, one_length)
        self.assertFalse(collapsed.identified)
        self.assertIn("没有分开", collapsed.reason)

    def test_both_sides_split_scales_with_degree_squared(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        load = default_load()
        split = StageSplit("cp_qk", 4)
        report = SplitReport("cp_qk", 0.0, 0.0, 0.0, 0.0, False, True, (), 0.0, None, "")
        estimate = predict_stage(spec, complexity, load, split, report)
        self.assertAlmostEqual(estimate.speedup, 16.0, places=4)
        self.assertTrue(math.isinf(estimate.speedup_upper_bound))

    def test_overlapping_communication_is_not_added(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        loads = [default_load().varied(length=length) for length in (128, 512, 4096)]
        observations = []
        for degree in (2, 8):
            observations.extend(
                measure(
                    spec,
                    coefficients,
                    loads,
                    StageSplit("cp_q", degree),
                    kappa=0.4,
                    tau_s=0.05,
                    v_s_per_byte=1e-6,
                    overlap=True,
                    uncovered_s=0.01,
                )
            )
        report = fit_split(spec, complexity, observations)
        self.assertTrue(report.overlap)
        load = loads[1]
        estimate = predict_stage(spec, complexity, load, StageSplit("cp_q", 2), report)
        truth, _, _ = stage_time(
            spec,
            coefficients,
            load,
            StageSplit("cp_q", 2),
            kappa=0.4,
            tau_s=0.05,
            v_s_per_byte=1e-6,
            overlap=True,
            uncovered_s=0.01,
        )
        assert_rel(self, estimate.time_s, truth)

    def test_mismatched_bytes_are_not_treated_as_identified(self) -> None:
        spec = StageSpec("attn", "self_attn")
        coefficients = {"self_attn": 2e-9}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        observations = measure(
            spec,
            coefficients,
            [default_load(), default_load().varied(length=2048)],
            StageSplit("cp_q", 4),
            kappa=0.2,
            tau_s=0.01,
            v_s_per_byte=1e-6,
        )
        forged = [
            StageObservation(obs.load, obs.split, obs.time_s, (obs.comm_bytes or 0.0) * 3.0, obs.comm_time_s)
            for obs in observations
        ]
        report = fit_split(spec, complexity, forged)
        self.assertFalse(report.identified)
        self.assertFalse(report.volume_matches)

    def test_pipeline_scales_by_the_bubble_factor(self) -> None:
        spec = StageSpec("proj", "linear")
        coefficients = {"linear": 1e-8}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        loads = [default_load(), default_load().varied(length=1024)]
        split = StageSplit("pp", 4, 8)
        report = fit_split(
            spec,
            complexity,
            measure(spec, coefficients, loads, split, tau_s=0.002, v_s_per_byte=1e-8),
        )
        self.assertIn("kappa", report.inactive)
        assert_rel(self, report.tau_s, 0.002)
        assert_rel(self, report.v_s_per_byte, 1e-8)
        estimate = predict_stage(spec, complexity, loads[0], split, report)
        self.assertAlmostEqual(estimate.speedup_upper_bound, 4.0)


class NetworkTests(unittest.TestCase):
    def test_query_split_does_not_shrink_the_projection(self) -> None:
        embed, proj, attn = hstu_stages()
        embed_coef = {"lookup": 1e-5, "intercept": 0.01}
        proj_coef = {"linear": 1e-8}
        attn_coef = {"self_attn": 2e-9}
        complexities = {
            embed.name: fit_complexity(embed, measure(embed, embed_coef, complexity_grid(embed), StageSplit("none"))),
            proj.name: fit_complexity(proj, measure(proj, proj_coef, complexity_grid(proj), StageSplit("none"))),
            attn.name: fit_complexity(attn, measure(attn, attn_coef, complexity_grid(attn), StageSplit("none"))),
        }
        load = default_load()
        split = StageSplit("cp_q", 4)
        split_fit = SplitReport("cp_q", 0.0, 0.0, 0.0, 0.0, False, True, (), 0.0, None, "")
        network = predict_network(
            (embed, proj, attn),
            complexities,
            {attn.name: split_fit},
            load,
            {attn.name: split},
        )
        embed_time, _, _ = stage_time(embed, embed_coef, load, StageSplit("none"))
        proj_time, _, _ = stage_time(proj, proj_coef, load, StageSplit("none"))
        attn_time, _, _ = stage_time(attn, attn_coef, load, split)
        assert_rel(self, network.time_s, embed_time + proj_time + attn_time)
        self.assertAlmostEqual(network.stages[2][1].speedup, 4.0, places=4)
        self.assertEqual(network.stages[2][1].split.kind, "cp_q")
        self.assertEqual(network.stages[0][1].split.kind, "none")

    def test_fitted_split_is_used_when_the_plan_says_none(self) -> None:
        embed, proj, attn = hstu_stages()
        attn_coef = {"self_attn": 2e-9, "intercept": 0.02}
        complexities = {
            embed.name: fit_complexity(
                embed,
                measure(embed, {"lookup": 1e-5, "intercept": 0.01}, complexity_grid(embed), StageSplit("none")),
            ),
            proj.name: fit_complexity(
                proj, measure(proj, {"linear": 1e-8}, complexity_grid(proj), StageSplit("none"))
            ),
            attn.name: fit_complexity(
                attn, measure(attn, attn_coef, complexity_grid(attn), StageSplit("none"))
            ),
        }
        loads = [default_load().varied(length=length) for length in (512, 2048)]
        fitted = fit_split(
            attn,
            complexities[attn.name],
            measure(
                attn,
                attn_coef,
                loads,
                StageSplit("cp_q", 4),
                kappa=0.25,
                tau_s=0.01,
                v_s_per_byte=1e-6,
            ),
        )
        self.assertEqual(fitted.degree, 4)
        load = default_load()
        network = predict_network((embed, proj, attn), complexities, {attn.name: fitted}, load)
        self.assertEqual([piece.split.kind for _, piece in network.stages], ["none", "none", "cp_q"])
        self.assertEqual(network.stages[2][1].split.degree, 4)
        overridden = predict_network(
            (embed, proj, attn),
            complexities,
            {attn.name: fitted},
            load,
            {attn.name: StageSplit("none")},
        )
        self.assertEqual(overridden.stages[2][1].split.kind, "cp_q")
        mixed = []
        for degree in (2, 4):
            mixed.extend(
                measure(
                    attn,
                    attn_coef,
                    loads,
                    StageSplit("cp_q", degree),
                    kappa=0.25,
                    tau_s=0.01,
                    v_s_per_byte=1e-6,
                )
            )
        several = fit_split(attn, complexities[attn.name], mixed)
        self.assertIsNone(several.degree)
        with self.assertRaises(FrameworkError):
            predict_network((embed, proj, attn), complexities, {attn.name: several}, load)
        explicit = predict_network(
            (embed, proj, attn),
            complexities,
            {attn.name: several},
            load,
            {attn.name: StageSplit("cp_q", 4)},
        )
        self.assertEqual(explicit.stages[2][1].split.kind, "cp_q")
        self.assertEqual(explicit.stages[2][1].split.degree, 4)

    def test_cross_attention_shards_the_key_length(self) -> None:
        spec = StageSpec("decode", "cross_attn")
        coefficients = {"cross_attn": 1e-8}
        complexity = fit_complexity(spec, measure(spec, coefficients, complexity_grid(spec), StageSplit("none")))
        load = default_load()
        report = SplitReport("shard_kv", 0.0, 0.0, 0.0, 0.0, False, True, (), 0.0, None, "")
        estimate = predict_stage(spec, complexity, load, StageSplit("shard_kv", 4), report)
        self.assertAlmostEqual(estimate.speedup, 4.0, places=4)

    def test_hyformer_decode_and_sequence_cross_read_different_lengths(self) -> None:
        stages = {spec.name: spec for spec in hyformer_stages("longer")}
        load = default_load().varied(extras={"short_len": 8, "decode_queries": 3, "boost_tokens": 16})
        sequence = features(stages["seq_cross"], load)["cross_attn"]
        decode = features(stages["decode"], load)["cross_attn"]
        self.assertNotEqual(sequence, decode)
        self.assertEqual(sequence, load.batch * 8 * load.length * load.width)
        self.assertEqual(decode, load.batch * 3 * 8 * load.width)
        self.assertTrue(longer_stages()[1].groups_follow_length)
        self.assertEqual(rankmixer_stages()[0].operator, "token_permute")

    def test_csv_roundtrip(self) -> None:
        spec = StageSpec("attn", "self_attn", axes={"length": "short_len"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "阶段.csv"
            write_stage_template(path, spec)
            with path.open(encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            self.assertIn("x_short_len", header)
            row = {name: "" for name in header}
            row.update(
                {
                    "batch": "2",
                    "length": "512",
                    "n_id": "128",
                    "n_query": "8",
                    "length_kv": "512",
                    "tokens": "16",
                    "groups": "4",
                    "width": "64",
                    "hidden": "256",
                    "n_head": "4",
                    "d_head": "16",
                    "split": "none",
                    "time_s": "0.2",
                    "x_short_len": "32",
                }
            )
            with path.open("a", encoding="utf-8") as handle:
                handle.write(",".join(row[name] for name in header) + "\n")
            loaded = load_stage_samples(path)
        self.assertEqual(loaded[0].load.extras["short_len"], 32)
        self.assertEqual(loaded[0].split.kind, "none")


if __name__ == "__main__":
    unittest.main()
