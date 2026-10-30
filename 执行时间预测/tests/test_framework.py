"""合成样本上的公式、拟合和留出。"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from timepred.baseline import BlockBaseline, features
from timepred.evaluate import evaluate
from timepred.fit import fit
from timepred.formula import compute_speedup, linear_cut, slide_time
from timepred.holdout import holdout_by_load
from timepred.io import load_samples, write_template
from timepred.linalg import fit_relative_linear, nnls
from timepred.memory import memory_parts
from timepred.predict import SchemeParams, TimeModel
from timepred.schema import (
    BlockSpec,
    FrameworkError,
    Load,
    ModelSpec,
    Plan,
    Sample,
    Split,
    active_kinds,
)
from timepred.synthetic import build_problem, make_load, make_spec


def assert_rel(test: unittest.TestCase, got: float, expected: float) -> None:
    scale = max(abs(expected), 1e-30)
    test.assertLess(
        abs(got - expected),
        1e-4 * scale + 1e-12,
        msg=f"{got} != {expected}",
    )


class ComplexityTests(unittest.TestCase):
    def test_token_mixer_uses_groups_and_tokens_not_sequence_length(self) -> None:
        block = BlockSpec("mixer", "token_mixer", d_model=4)
        short = Load(2, {"mixer": 8, "attn": 512}, n_groups={"mixer": 3})
        long = Load(2, {"mixer": 8, "attn": 8192}, n_groups={"mixer": 3})
        self.assertEqual(features(block, short), [2 * 3 * 8 * 8 * 4, 1.0])
        self.assertEqual(features(block, short), features(block, long))

    def test_context_parallel_does_not_apply_to_token_mixer(self) -> None:
        spec = ModelSpec(
            blocks=(BlockSpec("mixer", "token_mixer", d_model=4),),
            capacity_bytes=10**9,
        )
        model = TimeModel(
            spec,
            {"mixer": BlockBaseline((1e-6, 0.0))},
            {},
        )
        load = Load(1, {"mixer": 16}, n_groups={"mixer": 2})
        with self.assertRaises(FrameworkError):
            model.predict(load, Plan({"mixer": Split("cp", 2)}))


class FormulaTests(unittest.TestCase):
    def test_nnls_recovers_nonnegative_coefficients(self) -> None:
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
        coef = nnls(rows, [2.0, 0.0, 3.0, 5.0])
        self.assertAlmostEqual(coef[0], 2.0, places=6)
        self.assertAlmostEqual(coef[1], 0.0, places=6)
        self.assertAlmostEqual(coef[2], 3.0, places=6)

    def test_long_rows_keep_the_square_coefficient_when_their_weight_is_small(self) -> None:
        coefficient = 2e-9
        intercept = 0.02
        rows: list[list[float]] = []
        measured: list[float] = []
        for length in (128, 256, 512):
            for batch in (1, 2, 4, 8):
                feature = float(batch * 4 * length * length * 16)
                for _repeat in range(4):
                    rows.append([feature, 1.0])
                    measured.append(coefficient * feature + intercept)
        for length in (8192, 16384):
            feature = float(4 * 4 * length * length * 16)
            rows.append([feature, 1.0])
            measured.append(coefficient * feature + intercept)
        weights = [1.0 / time for time in measured]
        long_share = sum(weights[-2:]) / sum(weights)
        self.assertLess(long_share, 0.01)
        theta = fit_relative_linear(rows, [0.0] * len(rows), measured)
        assert_rel(self, theta[0], coefficient)
        assert_rel(self, theta[1], intercept)
        peak = float(8 * 8 * (32768 ** 2) * 64)
        assert_rel(self, theta[0] * peak + theta[1], coefficient * peak + intercept)

    def test_linear_cut_matches_speedup(self) -> None:
        t0 = 3.45
        kappa = 0.35
        degree = 8
        offset, feature = linear_cut(t0, "cp", degree, 1)
        g = compute_speedup(kappa, degree)
        self.assertAlmostEqual(offset + kappa * feature, t0 / g, places=12)

    def test_single_scheme_matches_slide_formula(self) -> None:
        problem = build_problem()
        load = make_load(1, 8192)
        breakdown = problem.truth.predict(load, problem.plans["cp8"])
        recon = slide_time(breakdown.t0_s, breakdown.r, breakdown.g, breakdown.c)
        self.assertAlmostEqual(breakdown.time_s, recon, places=9)
        self.assertAlmostEqual(sum(breakdown.block_time_s.values()), breakdown.time_s, places=9)
        self.assertLessEqual(breakdown.speedup, breakdown.speedup_upper_bound + 1e-9)

    def test_short_sequence_slower_than_long_sequence(self) -> None:
        problem = build_problem()
        short = problem.truth.predict(make_load(1, 512), problem.plans["cp8"])
        long = problem.truth.predict(make_load(1, 8192), problem.plans["cp8"])
        self.assertLess(short.speedup, 1.0)
        self.assertGreater(long.speedup, 1.0)

    def test_mixed_time_is_sum_of_block_times(self) -> None:
        problem = build_problem()
        load = make_load(1, 4096)
        t0 = problem.truth.predict(load, problem.plans["none"]).time_s
        cp = problem.truth.predict(load, problem.plans["cp8"]).time_s
        emb = problem.truth.predict(load, problem.plans["embp"]).time_s
        mixed = problem.truth.predict(load, problem.plans["mixed"]).time_s
        self.assertAlmostEqual(mixed, cp + emb - t0, places=9)

    def test_dp_keeps_single_forward_time(self) -> None:
        problem = build_problem()
        load = make_load(10, 512)
        none = problem.truth.predict(load, problem.plans["none"])
        dp = problem.truth.predict(load, problem.plans["dp"])
        self.assertAlmostEqual(dp.time_s, none.time_s, places=12)

    def test_cp_does_not_shrink_kv(self) -> None:
        problem = build_problem()
        load = make_load(1, 4096)
        plain = memory_parts(problem.spec, load, problem.plans["none"])
        cp = memory_parts(problem.spec, load, problem.plans["cp8"])
        tp = memory_parts(problem.spec, load, problem.plans["tp"])
        self.assertEqual(cp.kv_bytes, plain.kv_bytes)
        self.assertLess(cp.activation_bytes, plain.activation_bytes)
        self.assertLess(tp.kv_bytes, plain.kv_bytes)


class FitTests(unittest.TestCase):
    def test_roundtrip_on_noiseless_samples(self) -> None:
        problem = build_problem()
        report = fit(problem.spec, problem.samples)
        for kind, truth in problem.truth.schemes.items():
            got = report.model.schemes[kind]
            assert_rel(self, got.kappa, truth.kappa)
            assert_rel(self, got.tau_s, truth.tau_s)
            assert_rel(self, got.v_s_per_byte, truth.v_s_per_byte)
        self.assertIn("kappa", report.inactive_parameters["pp"])
        self.assertLess(report.train_mean_abs_log_error, 1e-6)

    def test_holdout_keeps_a_load_together_and_does_not_update_on_test(self) -> None:
        problem = build_problem()
        held = holdout_by_load(problem.samples, test_fraction=0.2, seed=0)
        self.assertEqual(len(held.test_keys), 1)
        train_keys = {sample.load.key() for sample in held.train}
        test_keys = {sample.load.key() for sample in held.test}
        self.assertTrue(test_keys.isdisjoint(train_keys))
        report = fit(problem.spec, held.train)
        params = report.model.schemes["cp"]
        checked = evaluate(report.model, held.test)
        self.assertIs(report.model.schemes["cp"], params)
        self.assertLess(checked.mape, 1e-4)
        self.assertEqual(checked.rank_accuracy, 1.0)

    def test_mixed_samples_do_not_change_fitted_params(self) -> None:
        problem = build_problem()
        clean = []
        dirty = []
        for sample in problem.samples:
            if len(active_kinds(problem.spec, sample.plan)) > 1:
                dirty.append(
                    Sample(sample.load, sample.plan, sample.time_s * 10.0, None, sample.peak_mem_bytes)
                )
            else:
                clean.append(sample)
                dirty.append(sample)
        left = fit(problem.spec, clean)
        right = fit(problem.spec, dirty)
        for kind in left.model.schemes:
            assert_rel(self, right.model.schemes[kind].kappa, left.model.schemes[kind].kappa)
            assert_rel(self, right.model.schemes[kind].tau_s, left.model.schemes[kind].tau_s)
            assert_rel(self, right.model.schemes[kind].v_s_per_byte, left.model.schemes[kind].v_s_per_byte)

    def test_selects_feasible_plan_with_shortest_predicted_time(self) -> None:
        problem = build_problem()
        short = make_load(1, 512)
        long = make_load(1, 8192)
        self.assertEqual(
            problem.truth.select(short, [problem.plans["none"], problem.plans["cp8"]]),
            problem.plans["none"],
        )
        self.assertEqual(
            problem.truth.select(long, [problem.plans["none"], problem.plans["cp8"]]),
            problem.plans["cp8"],
        )
        plain = memory_parts(problem.spec, short, problem.plans["none"]).total_bytes
        sharded = memory_parts(problem.spec, short, problem.plans["embp"]).total_bytes
        self.assertLess(sharded, plain)
        tight = TimeModel(
            make_spec((plain + sharded) // 2),
            problem.truth.baselines,
            problem.truth.schemes,
        )
        chosen = tight.select(short, [problem.plans["none"], problem.plans["embp"]])
        self.assertEqual(chosen, problem.plans["embp"])

    def test_rejects_split_on_the_wrong_block(self) -> None:
        problem = build_problem()
        with self.assertRaises(FrameworkError):
            problem.truth.predict(make_load(1, 512), Plan({"embedding": Split("cp", 2)}))

    def test_expert_parallel_roundtrip(self) -> None:
        spec = ModelSpec(
            blocks=(BlockSpec("expert", "expert", d_model=32, n_layers=2, param_bytes=4096),),
            capacity_bytes=10**9,
        )
        truth = TimeModel(
            spec,
            {"expert": BlockBaseline((1e-6, 1e-3))},
            {"ep": SchemeParams(0.25, 1e-4, 1e-9)},
        )
        samples = []
        for batch, length, share in ((1, 128, 0.5), (2, 256, 0.5), (4, 512, 1.0), (1, 1024, 0.25)):
            load = Load(batch, {"expert": length}, expert_frac=share)
            plain = truth.predict(load, Plan({}))
            samples.append(Sample(load, Plan({}), plain.time_s, plain.block_t0_s))
            plan = Plan({"expert": Split("ep", 4)})
            samples.append(Sample(load, plan, truth.predict(load, plan).time_s))
        report = fit(spec, samples)
        assert_rel(self, report.model.schemes["ep"].kappa, 0.25)
        assert_rel(self, report.model.schemes["ep"].tau_s, 1e-4)
        assert_rel(self, report.model.schemes["ep"].v_s_per_byte, 1e-9)

    def test_csv_roundtrip(self) -> None:
        problem = build_problem()
        sample = problem.samples[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "测量.csv"
            write_template(path, problem.spec)
            with path.open(encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
            row = {name: "" for name in header}
            row["batch"] = str(sample.load.batch)
            row["time_s"] = repr(sample.time_s)
            row["plan"] = "none"
            row["expert_frac"] = "1"
            for name, length in sample.load.lengths.items():
                row[f"len_{name}"] = str(length)
            for name, count in sample.load.n_ids.items():
                row[f"ids_{name}"] = str(count)
            for name, segment in sample.block_times_s.items():
                row[f"t_{name}"] = repr(segment)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(",".join(row[name] for name in header) + "\n")
            loaded = load_samples(path, problem.spec)
        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(loaded[0].time_s, sample.time_s, places=9)
        self.assertEqual(loaded[0].load.lengths["attention"], sample.load.lengths["attention"])


class UpperBoundTests(unittest.TestCase):
    def test_fully_split_network_has_no_finite_upper_bound(self) -> None:
        spec = ModelSpec(
            blocks=(BlockSpec("attention", "attention", d_model=16, n_layers=1),),
            capacity_bytes=10**9,
        )
        model = TimeModel(
            spec,
            {"attention": BlockBaseline((0.0, 1e-6, 0.0))},
            {"cp": SchemeParams(0.0, 0.0, 0.0)},
        )
        breakdown = model.predict(Load(1, {"attention": 32}), Plan({"attention": Split("cp", 2)}))
        self.assertTrue(math.isinf(breakdown.speedup_upper_bound))


if __name__ == "__main__":
    unittest.main()
