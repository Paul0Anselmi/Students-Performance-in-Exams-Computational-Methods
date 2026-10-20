"""Confere os testes paramétricos contra fórmulas feitas à mão."""

import numpy as np

from studperf.stats.parametric import one_way_anova, paired_t_test, welch_t_test


def test_welch_matches_formula() -> None:
    rng = np.random.default_rng(0)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 2, 60)
    res = welch_t_test(x, y)
    t = (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / 40 + y.var(ddof=1) / 60)
    assert np.isclose(res.statistic, t)
    assert res.ci[0] < res.estimate < res.ci[1]


def test_paired_t_matches_formula() -> None:
    z = np.array([0.5, -0.2, 1.1, 0.8, 0.3])
    res = paired_t_test(z)
    assert np.isclose(res.statistic, z.mean() / (z.std(ddof=1) / np.sqrt(5)))
    assert res.df == (4.0,)


def test_anova_matches_formula() -> None:
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9])
    g = np.repeat(np.arange(3), 3)
    res = one_way_anova(y, g)
    ssb = 3 * ((np.array([2.0, 5, 8]) - 5) ** 2).sum()
    ssw = 3 * 2.0
    assert np.isclose(res.statistic, (ssb / 2) / (ssw / 6))
