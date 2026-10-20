r"""Bootstrap não paramétrico (Efron, 1979), tradução vetorizada de ``metodos de reamostragem.R``.

Princípio *plug-in*: a distribuição empírica
:math:`\hat F(x) = \#\{y_i \le x\}/n` substitui a distribuição desconhecida
:math:`F`. Reamostrar :math:`n` valores com reposição dos dados equivale a
sortear uma amostra iid de :math:`\hat F`; repetindo :math:`B` vezes obtemos
:math:`\hat\theta^*_1,\dots,\hat\theta^*_B`, que aproximam a distribuição
amostral de :math:`\hat\theta`.

Quantidades derivadas:

* erro padrão bootstrap
  :math:`\widehat{EP}_B = \sqrt{\tfrac{1}{B-1}\sum_b(\hat\theta^*_b-\hat\theta^*_{(\cdot)})^2}`;
* vício :math:`\widehat{\text{vício}} = \hat\theta^*_{(\cdot)} - \hat\theta` e
  estimador corrigido :math:`\tilde\theta = 2\hat\theta - \hat\theta^*_{(\cdot)}`;
* IC percentil :math:`(\hat F^{*-1}(\alpha/2),\ \hat F^{*-1}(1-\alpha/2))` e IC
  normal :math:`\hat\theta \pm z_{1-\alpha/2}\widehat{EP}_B`.

As reamostras são geradas em blocos ``(chunk, n)`` de índices e a estatística
é avaliada por linha (``axis=1``), sem laços sobre :math:`b`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from studperf.stats.permutation import _chunks

FloatArray = NDArray[np.float64]
#: Estatística vetorizada: recebe matriz ``(B, n)`` e devolve vetor ``(B,)``.
RowStatistic = Callable[[FloatArray], FloatArray]


def row_mean(a: FloatArray) -> FloatArray:
    """Média por linha."""
    return a.mean(axis=1)


def row_median(a: FloatArray) -> FloatArray:
    """Mediana por linha."""
    return np.median(a, axis=1)


@dataclass(frozen=True)
class BootstrapResult:
    """Resumo de uma distribuição bootstrap.

    Attributes:
        estimate: :math:`\\hat\\theta` na amostra original.
        replicates: :math:`\\hat\\theta^*_b`, ``b = 1..B``.
        se: erro padrão bootstrap.
        bias: vício bootstrap.
        bias_corrected: :math:`2\\hat\\theta - \\hat\\theta^*_{(\\cdot)}`.
        ci_percentile: IC percentil.
        ci_normal: IC supondo normalidade de :math:`\\hat\\theta`.
        level: coeficiente de confiança :math:`\\gamma`.
    """

    estimate: float
    replicates: FloatArray = field(repr=False)
    se: float
    bias: float
    bias_corrected: float
    ci_percentile: tuple[float, float]
    ci_normal: tuple[float, float]
    level: float


def summarize(estimate: float, replicates: FloatArray, level: float = 0.95) -> BootstrapResult:
    """Calcula EP, vício e ICs a partir das réplicas bootstrap."""
    alpha = 1 - level
    se = float(replicates.std(ddof=1))
    mean_star = float(replicates.mean())
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    z = float(norm.ppf(1 - alpha / 2))
    return BootstrapResult(
        estimate=estimate,
        replicates=replicates,
        se=se,
        bias=mean_star - estimate,
        bias_corrected=2 * estimate - mean_star,
        ci_percentile=(float(lo), float(hi)),
        ci_normal=(estimate - z * se, estimate + z * se),
        level=level,
    )


def bootstrap_replicates(
    data: FloatArray,
    stat: RowStatistic,
    *,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
    chunk: int = 2_000,
) -> FloatArray:
    """Réplicas :math:`\\hat\\theta^*_b` para uma amostra univariada."""
    rng = rng or np.random.default_rng()
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    return np.concatenate([stat(data[rng.integers(0, n, size=(c, n))]) for c in _chunks(n_resamples, chunk)])


def bootstrap_one_sample(
    data: FloatArray,
    stat: RowStatistic = row_mean,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Bootstrap de uma estatística de uma amostra (ex.: média das diferenças pareadas)."""
    data = np.asarray(data, dtype=np.float64)
    est = float(stat(data[None, :])[0])
    return summarize(est, bootstrap_replicates(data, stat, n_resamples=n_resamples, rng=rng), level)


def bootstrap_two_sample(
    x: FloatArray,
    y: FloatArray,
    stat: RowStatistic = row_mean,
    *,
    n_resamples: int = 10_000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    r"""IC bootstrap para :math:`\theta = \theta(F) - \theta(G)` com amostras independentes.

    ``x`` e ``y`` são reamostrados separadamente (cada um de sua própria
    :math:`\hat F`, :math:`\hat G`), preservando os tamanhos :math:`n` e :math:`m`.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    est = float(stat(x[None, :])[0] - stat(y[None, :])[0])
    rep = bootstrap_replicates(x, stat, n_resamples=n_resamples, rng=rng) - bootstrap_replicates(
        y, stat, n_resamples=n_resamples, rng=rng
    )
    return summarize(est, rep, level)


@dataclass(frozen=True)
class BootstrapTestResult:
    """Teste de hipóteses bootstrap: estatística observada, réplicas sob H0 e valor-p."""

    statistic: float
    p_value: float
    null_distribution: FloatArray = field(repr=False)


def _welch_t(x: FloatArray, y: FloatArray) -> FloatArray:
    """Estatística t de Welch por linha (variâncias possivelmente diferentes)."""
    n, m = x.shape[1], y.shape[1]
    return (x.mean(axis=1) - y.mean(axis=1)) / np.sqrt(x.var(axis=1, ddof=1) / n + y.var(axis=1, ddof=1) / m)


def bootstrap_two_mean_test(
    x: FloatArray,
    y: FloatArray,
    *,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> BootstrapTestResult:
    r"""Teste bootstrap de :math:`H_0:\mu_F=\mu_G` (bilateral) sem supor variâncias iguais.

    :math:`\hat F_0` é obtida transladando cada amostra para a média combinada
    :math:`\bar z`: :math:`\tilde x_i = x_i-\bar x+\bar z`,
    :math:`\tilde y_j = y_j-\bar y+\bar z`. Valor-p:
    :math:`\#\{|T(z^*_b)|\ge|T(z)|\}/B`. Complementa o teste de permutação
    quando a suposição de distribuições idênticas sob :math:`H_0` é duvidosa.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    zbar = np.concatenate([x, y]).mean()
    xt, yt = x - x.mean() + zbar, y - y.mean() + zbar
    t_obs = float(_welch_t(x[None, :], y[None, :])[0])
    null = np.concatenate(
        [
            _welch_t(xt[rng.integers(0, x.size, (c, x.size))], yt[rng.integers(0, y.size, (c, y.size))])
            for c in _chunks(n_resamples, 2_000)
        ]
    )
    p = float(np.count_nonzero(np.abs(null) >= abs(t_obs)) / null.size)
    return BootstrapTestResult(t_obs, p, null)


# ---------------------------------------------------------------------------
# Regressão linear
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OLSFit:
    r"""Ajuste por mínimos quadrados :math:`\hat\beta=(X^TX)^{-1}X^Ty`."""

    beta: FloatArray
    se_classic: FloatArray
    fitted: FloatArray
    residuals: FloatArray
    r2: float


def ols(X: FloatArray, y: FloatArray) -> OLSFit:
    """MQO com erro padrão clássico :math:`\\sqrt{\\hat\\sigma^2 \\mathrm{diag}(X^TX)^{-1}}`."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    fitted = X @ beta
    resid = y - fitted
    n, p = X.shape
    sigma2 = float(resid @ resid / (n - p))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return OLSFit(beta, np.sqrt(sigma2 * np.diag(xtx_inv)), fitted, resid, r2)


def bootstrap_regression_residuals(
    X: FloatArray, y: FloatArray, *, n_resamples: int = 5_000, rng: np.random.Generator | None = None
) -> FloatArray:
    r"""Bootstrap dos resíduos: :math:`y^*_b = X\hat\beta + e^*_b` com :math:`X` fixo.

    Supõe modelo linear correto e erros iid (homocedásticos). Como
    :math:`A=(X^TX)^{-1}X^T` é fixo, todas as réplicas saem de um único produto
    ``Y_star @ A.T``. Retorna matriz ``(B, p)``.
    """
    rng = rng or np.random.default_rng()
    fit = ols(X, y)
    a = np.linalg.solve(X.T @ X, X.T)
    n = y.size
    return np.concatenate(
        [(fit.fitted + fit.residuals[rng.integers(0, n, (c, n))]) @ a.T for c in _chunks(n_resamples, 1_000)]
    )


def bootstrap_regression_pairs(
    X: FloatArray, y: FloatArray, *, n_resamples: int = 5_000, rng: np.random.Generator | None = None
) -> FloatArray:
    r"""Bootstrap dos pares :math:`(x_i, y_i)`.

    Única suposição: os pares são uma amostra aleatória de uma distribuição
    conjunta :math:`G`. É robusto a heterocedasticidade e a má especificação.
    Cada bloco resolve :math:`B_c` sistemas :math:`(X^{*T}X^*)\beta=X^{*T}y^*`
    de uma vez com ``einsum`` + ``np.linalg.solve`` em lote.
    """
    rng = rng or np.random.default_rng()
    n = y.size
    out: list[FloatArray] = []
    for c in _chunks(n_resamples, 250):
        idx = rng.integers(0, n, (c, n))
        xs, ys = X[idx], y[idx]
        xtx = np.einsum("bni,bnj->bij", xs, xs)
        xty = np.einsum("bni,bn->bi", xs, ys)
        out.append(np.linalg.solve(xtx, xty[..., None])[..., 0])
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Escolha do número B de reamostras
# ---------------------------------------------------------------------------

#: Gera ``B`` réplicas bootstrap com o gerador dado: ``(B, rng) -> (B,)``.
ReplicateFn = Callable[[int, np.random.Generator], FloatArray]


@dataclass(frozen=True)
class BChoice:
    """Variação de Monte Carlo do bootstrap para cada B testado.

    Para cada ``B`` o bootstrap é repetido ``repeats`` vezes com sementes
    diferentes. Se o resultado muda pouco entre repetições, B é suficiente.

    Attributes:
        b_values: valores de B testados.
        se_mean: média do EP bootstrap entre as repetições.
        se_cv: coeficiente de variação do EP (desvio / média).
        ci_lo_sd: desvio padrão do limite inferior do IC percentil.
        ci_hi_sd: desvio padrão do limite superior do IC percentil.
        chosen: menor B com ``se_cv`` e desvios dos limites abaixo das tolerâncias.
    """

    b_values: NDArray[np.int64]
    se_mean: FloatArray
    se_cv: FloatArray
    ci_lo_sd: FloatArray
    ci_hi_sd: FloatArray
    chosen: int


def choose_b(
    replicate: ReplicateFn,
    b_values: tuple[int, ...] = (200, 500, 1_000, 2_000, 5_000, 10_000, 20_000),
    *,
    repeats: int = 30,
    level: float = 0.95,
    se_tol: float = 0.01,
    ci_tol: float = 0.05,
    rng: np.random.Generator | None = None,
) -> BChoice:
    r"""Roda o bootstrap para vários B e escolhe o menor B estável.

    O bootstrap tem duas fontes de erro: a amostral (fixa, vem dos dados) e a
    de Monte Carlo (cai com :math:`1/\sqrt B`). B é suficiente quando a de
    Monte Carlo fica desprezível: o EP varia menos de ``se_tol`` (1%) entre
    repetições e os limites do IC variam menos de ``ci_tol`` pontos de nota.
    """
    rng = rng or np.random.default_rng()
    alpha = 1 - level
    b_arr = np.asarray(b_values, dtype=np.int64)
    se_mean, se_cv, lo_sd, hi_sd = (np.empty(b_arr.size) for _ in range(4))
    for i, b in enumerate(b_arr):
        reps = np.stack([replicate(int(b), rng) for _ in range(repeats)])
        se = reps.std(axis=1, ddof=1)
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2], axis=1)
        se_mean[i], se_cv[i] = se.mean(), se.std(ddof=1) / se.mean()
        lo_sd[i], hi_sd[i] = lo.std(ddof=1), hi.std(ddof=1)
    ok = (se_cv < se_tol) & (lo_sd < ci_tol) & (hi_sd < ci_tol)
    chosen = int(b_arr[ok][0]) if ok.any() else int(b_arr[-1])
    return BChoice(b_arr, se_mean, se_cv, lo_sd, hi_sd, chosen)
