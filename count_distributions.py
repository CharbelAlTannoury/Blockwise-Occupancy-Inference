from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, norm

M = 20
P0 = .05
A_NULL = 3
K_RECORD = 6

def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:


    pmf = np.array([1.0])
    for p in np.asarray(probabilities, dtype=float):
        pmf = np.convolve(pmf, np.array([1.0 - p, p]))
    return pmf


def upper_tail_threshold_from_pmf(pmf: np.ndarray, tail_level: float) -> int:


    if not 0.0 < tail_level < 1.0:
        raise ValueError("tail_level must lie strictly between zero and one")
    for k in range(1, len(pmf)):
        if float(pmf[k:].sum()) <= tail_level:
            return k
    return len(pmf)


def iid_record_threshold(m: int, q_record: float) -> int:
    if m < 1:
        raise ValueError("m must be positive")
    probabilities = np.array([1.0 / j for j in range(1, m + 1)])
    return upper_tail_threshold_from_pmf(
        poisson_binomial_pmf(probabilities), q_record
    )


def yang_record_probabilities(m: int, gamma: float) -> np.ndarray:


    if m < 1 or gamma < 1 or not np.isfinite(gamma):
        raise ValueError("m must be positive and gamma finite and at least one")
    j = np.arange(1, m + 1, dtype=float)
    if gamma == 1:
        return 1 / j

    return -np.expm1(-np.log(gamma)) / -np.expm1(-j*np.log(gamma))


def markov_binomial_pmf(
    p0: float = P0,
    persistence: float = 0.70,
    m: int = M,
) -> np.ndarray:


    p11 = p0 + persistence * (1.0 - p0)
    p01 = p0 * (1.0 - persistence)
    transition = np.array([[1.0 - p01, p01], [1.0 - p11, p11]])


    dp = np.zeros((m + 1, 2))
    dp[0, 0] = 1.0 - p0
    dp[1, 1] = p0
    for step in range(1, m):
        updated = np.zeros_like(dp)
        for count in range(step + 1):
            for state in (0, 1):
                for next_state in (0, 1):
                    updated[count + next_state, next_state] += (
                        dp[count, state] * transition[state, next_state]
                    )
        dp = updated
    return dp.sum(axis=1)


def duration_strength_benchmark(
    m: int = M,
    p0: float = P0,
    a: int = A_NULL,
    k: int = K_RECORD,
    changed_fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    mean_shifts: tuple[float, ...] = (0.75, 1.25, 1.75, 2.25),
) -> pd.DataFrame:


    critical = float(norm.ppf(1.0 - p0 / 2.0))
    rows: list[dict[str, float | int]] = []
    for changed_fraction in changed_fractions:
        changed_count = int(round(m * changed_fraction))
        for mean_shift in mean_shifts:
            changed_tail = float(
                norm.sf(critical - mean_shift) + norm.cdf(-critical - mean_shift)
            )
            probabilities = np.r_[
                np.full(changed_count, changed_tail),
                np.full(m - changed_count, p0),
            ]
            pmf = poisson_binomial_pmf(probabilities)
            rows.append(
                {
                    "changed_fraction": changed_fraction,
                    "mean_shift": mean_shift,
                    "changed_exceedance_probability": changed_tail,
                    "mean_block_exceedance_probability": float(probabilities.mean()),
                    "probability_H0": float(pmf[: a + 1].sum()),
                    "probability_H1": float(pmf[a + 1 : k].sum()),
                    "probability_H2": float(pmf[k:].sum()),
                }
            )
    return pd.DataFrame(rows)


def probability_of_run(
    n: int,
    success_probability: float,
    run_length: int = 3,
) -> float:


    if n < 1 or run_length < 1:
        raise ValueError("n and run_length must be positive")
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must lie in [0, 1]")


    state = np.zeros(run_length)
    state[0] = 1.0
    for _ in range(n):
        updated = np.zeros_like(state)
        updated[0] = (1.0 - success_probability) * state.sum()
        updated[1:] = success_probability * state[:-1]
        state = updated
    return float(1.0 - state.sum())


def size_matched_gaussian_benchmark(
    m: int = M,
    p0: float = P0,
    k: int = K_RECORD,
) -> tuple[pd.DataFrame, pd.DataFrame]:


    alpha_persistent = float(binom.sf(k - 1, m, p0))
    p_run = float(
        brentq(
            lambda p: probability_of_run(m, p, run_length=3)
            - alpha_persistent,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
    )
    p_maximum = float(-np.expm1(np.log1p(-alpha_persistent) / m))

    critical_count = float(norm.ppf(1.0 - p0 / 2.0))
    critical_run = float(norm.ppf(1.0 - p_run / 2.0))
    critical_maximum = float(norm.ppf(1.0 - p_maximum / 2.0))
    critical_sum = float(norm.ppf(1.0 - alpha_persistent / 2.0))

    calibration = pd.DataFrame(
        [
            {
                "method": "record-calibrated count",
                "pointwise_tail": p0,
                "critical_value": critical_count,
                "exact_null_size": alpha_persistent,
            },
            {
                "method": "calibrated run of three",
                "pointwise_tail": p_run,
                "critical_value": critical_run,
                "exact_null_size": probability_of_run(m, p_run, run_length=3),
            },
            {
                "method": "calibrated maximum",
                "pointwise_tail": p_maximum,
                "critical_value": critical_maximum,
                "exact_null_size": float(
                    -np.expm1(m * np.log1p(-p_maximum))
                ),
            },
            {
                "method": "Gaussian block sum",
                "pointwise_tail": np.nan,
                "critical_value": critical_sum,
                "exact_null_size": float(2.0 * norm.sf(critical_sum)),
            },
        ]
    )

    alternatives = (
        (r"Null $\mathcal{N}(0,1)$", "mean", 0.0),
        (r"Mean $\mathcal{N}(0.60,1)$", "mean", 0.60),
        (r"Mean $\mathcal{N}(1.00,1)$", "mean", 1.00),
        (r"Scale $\mathcal{N}(0,1.25^2)$", "scale", 1.25),
        (r"Scale $\mathcal{N}(0,1.50^2)$", "scale", 1.50),
    )
    rows: list[dict[str, float | str]] = []

    for label, family, parameter in alternatives:
        if family == "mean":
            mu = parameter

            def two_sided_tail(critical: float) -> float:
                return float(
                    norm.sf(critical - mu) + norm.cdf(-critical - mu)
                )

            block_shift = math.sqrt(m) * mu
            block_scale = 1.0
        else:
            sigma = parameter

            def two_sided_tail(critical: float) -> float:
                return float(2.0 * norm.sf(critical / sigma))

            block_shift = 0.0
            block_scale = sigma

        p_count = two_sided_tail(critical_count)
        p_run_alt = two_sided_tail(critical_run)
        p_max_alt = two_sided_tail(critical_maximum)
        rows.append(
            {
                "alternative": label,
                "record_count": float(binom.sf(k - 1, m, p_count)),
                "calibrated_run3": probability_of_run(
                    m, p_run_alt, run_length=3
                ),
                "calibrated_maximum": float(
                    -np.expm1(m * np.log1p(-p_max_alt))
                ),
                "gaussian_block_sum": float(
                    norm.sf((critical_sum - block_shift) / block_scale)
                    + norm.cdf((-critical_sum - block_shift) / block_scale)
                ),
            }
        )

    return pd.DataFrame(rows), calibration
