r"""Testes de permutação (aleatorização), tradução vetorizada de ``teste de permutacao.R``.

Ideia (Fisher, 1935): sob :math:`H_0` as observações são *intercambiáveis*, logo
qualquer rearranjo dos dados é tão provável quanto o observado. A distribuição
de referência da estatística :math:`T` é obtida recalculando-a em todos (teste
exato) ou em :math:`B` rearranjos sorteados (teste de Monte Carlo).

Valor-p:

* exato: :math:`p = \#\{T^{(r)} \text{ tão ou mais extremo que } T_{obs}\} / R`;
* Monte Carlo: :math:`p = (1 + \#\{\cdot\}) / (B + 1)`, que inclui o arranjo
  observado e garante um teste de nível exato :math:`\alpha`
  (Phipson & Smyth, 2010).

Nenhuma função usa laço sobre observações: cada bloco de rearranjos é uma
matriz ``(chunk, n)`` processada com operações ``numpy``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Alternative = Literal["two-sided", "less", "greater"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PermutationResult:
    """Resultado de um teste de permutação.

    Attributes:
        statistic: valor observado :math:`T_{obs}`.
        p_value: valor-p (exato ou Monte Carlo).
        null_distribution: valores de :math:`T` sob :math:`H_0`.
        alternative: hipótese alternativa.
        exact: ``True`` se todas as permutações foram enumeradas.
        critical_values: quantis da região crítica para o ``alpha`` usado.
    """

    statistic: float
    p_value: float
    null_distribution: FloatArray = field(repr=False)
    alternative: Alternative
    exact: bool
    critical_values: tuple[float, ...]


def _chunks(total: int, size: int) -> Iterator[int]:
    """Tamanhos dos blocos para limitar memória (``size`` linhas por vez)."""
    full, rest = divmod(total, size)
    yield from [size] * full
    if rest:
        yield rest


def _p_value(null: FloatArray, t_obs: float, alternative: Alternative, exact: bool) -> float:
    """Valor-p como na apostila; bilateral = 2 × menor cauda (limitado a 1)."""
    tol = 1e-9 * max(1.0, abs(t_obs))
    le = np.count_nonzero(null <= t_obs + tol)
    ge = np.count_nonzero(null >= t_obs - tol)
    r = null.size
    if exact:
        lo, hi = le / r, ge / r
    else:
        lo, hi = (le + 1) / (r + 1), (ge + 1) / (r + 1)
    if alternative == "less":
        return float(lo)
    if alternative == "greater":
        return float(hi)
    return float(min(1.0, 2 * min(lo, hi)))


def _critical(null: FloatArray, alpha: float, alternative: Alternative) -> tuple[float, ...]:
    if alternative == "less":
        return (float(np.quantile(null, alpha)),)
    if alternative == "greater":
        return (float(np.quantile(null, 1 - alpha)),)
    return tuple(float(q) for q in np.quantile(null, [alpha / 2, 1 - alpha / 2]))


# ---------------------------------------------------------------------------
# Duas populações pareadas
# ---------------------------------------------------------------------------


def paired_permutation_test(
    diff: FloatArray,
    *,
    alternative: Alternative = "two-sided",
    n_resamples: int = 10_000,
    exact: bool | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    chunk: int = 2_000,
) -> PermutationResult:
    r"""Teste de permutação para duas populações pareadas (troca de sinais).

    Seja :math:`Z_i = X_i - Y_i` e :math:`M` a média populacional de :math:`Z`.
    Testa-se :math:`H_0: M = 0`. Sob :math:`H_0` não há efeito, então trocar
    :math:`X_i` com :math:`Y_i` dentro do par (isto é, trocar o sinal de
    :math:`Z_i`) não altera a distribuição. Há :math:`2^n` rearranjos e a
    estatística é :math:`T_1 = \bar Z`.

    Args:
        diff: diferenças :math:`z_i` de cada par.
        exact: enumera os :math:`2^n` sinais; padrão ``True`` se :math:`n \le 20`.
    """
    z = np.asarray(diff, dtype=np.float64)
    n = z.size
    t_obs = float(z.mean())
    exact = n <= 20 if exact is None else exact
    if exact:
        # Linha r da matriz: bits de r → sinais ±1 (equivale aos 9 laços do R).
        bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
        null = ((2 * bits - 1) * z).mean(axis=1)
    else:
        rng = rng or np.random.default_rng()
        parts = [(rng.choice((-1.0, 1.0), size=(c, n)) @ z) / n for c in _chunks(n_resamples, chunk)]
        null = np.concatenate(parts)
    return PermutationResult(
        t_obs, _p_value(null, t_obs, alternative, exact), null, alternative, exact, _critical(null, alpha, alternative)
    )


# ---------------------------------------------------------------------------
# Duas populações independentes
# ---------------------------------------------------------------------------

TwoSampleStat = Literal["sum_x", "mean_diff", "median_diff"]


def _two_sample_stat(pooled_perm: FloatArray, n: int, statistic: TwoSampleStat) -> FloatArray:
    """Estatística em cada linha de ``pooled_perm`` (primeiros ``n`` = grupo X)."""
    xs, ys = pooled_perm[:, :n], pooled_perm[:, n:]
    if statistic == "sum_x":
        return xs.sum(axis=1)
    if statistic == "mean_diff":
        return xs.mean(axis=1) - ys.mean(axis=1)
    return np.median(xs, axis=1) - np.median(ys, axis=1)


def two_sample_permutation_test(
    x: FloatArray,
    y: FloatArray,
    *,
    statistic: TwoSampleStat = "mean_diff",
    alternative: Alternative = "two-sided",
    n_resamples: int = 10_000,
    exact: bool | None = None,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    chunk: int = 1_000,
) -> PermutationResult:
    r"""Teste de permutação para duas populações independentes.

    Suposições: amostras aleatórias, independentes entre si e, sob
    :math:`H_0`, com a mesma distribuição (diferença apenas de locação sob
    :math:`H_1`). Então os :math:`n+m` valores são intercambiáveis e há
    :math:`\binom{n+m}{n}` maneiras de escolher a amostra X.

    Estatísticas: :math:`T_1=\sum X_i`, :math:`D_1=\bar X-\bar Y` ou
    :math:`D_2=\mathrm{Md}(X)-\mathrm{Md}(Y)`.

    Args:
        exact: enumera todas as combinações; padrão ``True`` se
            :math:`\binom{n+m}{n} \le 200\,000`.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pooled = np.concatenate([x, y])
    n, total = x.size, pooled.size
    t_obs = float(_two_sample_stat(pooled[None, :], n, statistic)[0])
    exact = comb(total, n) <= 200_000 if exact is None else exact
    if exact:
        idx_x = np.array(list(combinations(range(total), n)), dtype=np.int64)
        mask = np.ones((idx_x.shape[0], total), dtype=bool)
        np.put_along_axis(mask, idx_x, False, axis=1)
        idx_y = np.nonzero(mask)[1].reshape(idx_x.shape[0], total - n)
        null = _two_sample_stat(pooled[np.hstack([idx_x, idx_y])], n, statistic)
    else:
        rng = rng or np.random.default_rng()
        parts = [
            _two_sample_stat(pooled[rng.permuted(np.broadcast_to(np.arange(total), (c, total)), axis=1)], n, statistic)
            for c in _chunks(n_resamples, chunk)
        ]
        null = np.concatenate(parts)
    return PermutationResult(
        t_obs, _p_value(null, t_obs, alternative, exact), null, alternative, exact, _critical(null, alpha, alternative)
    )


# ---------------------------------------------------------------------------
# K > 2 populações independentes
# ---------------------------------------------------------------------------


def f_statistic(y_perm: FloatArray, onehot: FloatArray) -> FloatArray:
    r"""Estatística F da ANOVA de um fator, vetorizada por linha.

    .. math:: F = \frac{SQE/(K-1)}{SQD/(N-K)},\quad
              SQE = \sum_k S_k^2/n_k - N\bar y^2,\quad SQD = SQT - SQE,

    em que :math:`S_k` é a soma do grupo :math:`k`. Como :math:`SQT` é
    invariante a permutações, basta recalcular as somas por grupo, o que é um
    único produto matricial ``y_perm @ onehot``.
    """
    n_k = onehot.sum(axis=0)
    total, k = onehot.shape
    y0 = y_perm[0]
    sst = float(((y0 - y0.mean()) ** 2).sum())
    sums = y_perm @ onehot
    ssb = (sums**2 / n_k).sum(axis=1) - total * y0.mean() ** 2
    ssw = sst - ssb
    return (ssb / (k - 1)) / (ssw / (total - k))


def k_sample_permutation_test(
    y: FloatArray,
    groups: NDArray[np.int64],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    chunk: int = 1_000,
) -> PermutationResult:
    r"""ANOVA de um fator por permutação (K > 2 populações independentes).

    :math:`H_0: \mu_1=\dots=\mu_K` contra :math:`H_1`: ao menos uma média
    difere. Suposições: amostras aleatórias independentes, erros
    homocedásticos (mesma forma e dispersão), sem exigir normalidade. Os
    rótulos de grupo são permutados entre as :math:`N` observações e o
    valor-p é a frequência de :math:`F^{(b)} \ge F_{obs}`.
    """
    y = np.asarray(y, dtype=np.float64)
    k = int(groups.max()) + 1
    onehot = np.eye(k)[groups]
    total = y.size
    f_obs = float(f_statistic(y[None, :], onehot)[0])
    rng = rng or np.random.default_rng()
    parts = [
        f_statistic(rng.permuted(np.broadcast_to(y, (c, total)), axis=1), onehot) for c in _chunks(n_resamples, chunk)
    ]
    null = np.concatenate(parts)
    return PermutationResult(
        f_obs, _p_value(null, f_obs, "greater", False), null, "greater", False, _critical(null, alpha, "greater")
    )

