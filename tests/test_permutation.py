"""Valida a tradução Python contra os exemplos resolvidos na apostila (R)."""

import numpy as np

from studperf.stats.permutation import (
    k_sample_permutation_test,
    paired_permutation_test,
    two_sample_permutation_test,
)


def test_paired_exact_matches_lecture() -> None:
    x = np.array([1.83, 0.5, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30])
    y = np.array([0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29])
    res = paired_permutation_test(y - x, alternative="less")
    assert res.exact and res.null_distribution.size == 512
    assert round(res.statistic, 2) == -0.43
    assert round(res.p_value, 3) == 0.014


def test_two_sample_exact_matches_lecture() -> None:
    x = np.array([0, 1, 1, 0, -2.0])
    y = np.array([6, 7, 7, 4, -3, 9, 14.0])
    res = two_sample_permutation_test(x, y, statistic="sum_x")
    assert res.null_distribution.size == 792
    assert res.statistic == 0
    assert round(res.p_value, 3) == 0.028


def test_monte_carlo_close_to_exact() -> None:
    x = np.array([0, 1, 1, 0, -2.0])
    y = np.array([6, 7, 7, 4, -3, 9, 14.0])
    mc = two_sample_permutation_test(
        x, y, statistic="sum_x", exact=False, n_resamples=20_000, rng=np.random.default_rng(1)
    )
    assert abs(mc.p_value - 0.028) < 0.01


def test_k_sample_no_effect_is_uniformish() -> None:
    rng = np.random.default_rng(3)
    y = rng.exponential(size=60)
    g = np.repeat(np.arange(3), 20)
    res = k_sample_permutation_test(y, g, n_resamples=2_000, rng=rng)
    assert 0 < res.p_value <= 1

