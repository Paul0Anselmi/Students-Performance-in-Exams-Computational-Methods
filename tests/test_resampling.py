"""Propriedades básicas do bootstrap e do jackknife."""

import numpy as np

from studperf.stats.bootstrap import (
    bootstrap_one_sample,
    bootstrap_regression_pairs,
    bootstrap_regression_residuals,
    ols,
)
from studperf.stats.jackknife import jackknife_ols, leave_one_out_index


def test_bootstrap_mean_se_close_to_theory() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    res = bootstrap_one_sample(x, n_resamples=5_000, rng=rng)
    assert abs(res.se - x.std(ddof=1) / 20) < 0.005


def test_regression_octanagem_example() -> None:
    x = np.arange(1, 7.0)
    y = np.array([80.5, 81.6, 82.1, 83.7, 83.9, 85.0])
    X = np.column_stack([np.ones(6), x])
    fit = ols(X, y)
    rng = np.random.default_rng(100)
    res = bootstrap_regression_residuals(X, y, n_resamples=2_000, rng=rng)
    assert res.shape == (2_000, 2)
    assert abs(res[:, 1].mean() - fit.beta[1]) < 0.02
    pairs = bootstrap_regression_pairs(np.column_stack([np.ones(50), rng.normal(size=50)]), rng.normal(size=50), n_resamples=500, rng=rng)
    assert pairs.shape == (500, 2)


def test_jackknife_matches_explicit_refit() -> None:
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(30), rng.normal(size=30)])
    y = X @ np.array([1.0, 2.0]) + rng.normal(size=30)
    jk = jackknife_ols(X, y)
    idx = leave_one_out_index(30)
    explicit = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0] for i in idx])
    np.testing.assert_allclose(jk.replicates, explicit, atol=1e-10)


def test_choose_b_variation_falls_with_b() -> None:
    from studperf.stats.bootstrap import bootstrap_replicates, choose_b, row_mean

    x = np.random.default_rng(5).normal(size=200)
    res = choose_b(
        lambda b, rng: bootstrap_replicates(x, row_mean, n_resamples=b, rng=rng),
        (200, 2_000),
        repeats=20,
        rng=np.random.default_rng(6),
    )
    assert res.se_cv[1] < res.se_cv[0]
    assert res.chosen in (200, 2_000)
