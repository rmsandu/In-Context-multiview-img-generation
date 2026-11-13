"""Paired, prompt-clustered statistics for base-vs-LoRA comparisons.

Design (fixed in docs/study1_protocol.md before LoRA metrics were computed):

* The unit of analysis is the prompt. Generation seeds are repeated measurements
  within a prompt, not independent samples, so they are averaged within a prompt.
* Base and LoRA share prompts *and* seeds, so the comparison is paired at both
  levels. The bootstrap resamples prompts with replacement and, inside each
  resampled prompt, resamples seed indices with replacement, applying the same
  seed draw to both conditions to keep the pairing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


@dataclass(frozen=True)
class PairedEstimate:
    mean_a: float
    mean_b: float
    diff: float  # mean_b - mean_a
    ci_low: float
    ci_high: float
    n_prompts: int
    n_seeds: int
    n_boot: int

    def as_dict(self) -> dict:
        return asdict(self)


def _prompt_means(values: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanmean(values, axis=1)


def paired_cluster_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 17,
    confidence: float = 0.95,
) -> PairedEstimate:
    """Percentile CI for mean(b) - mean(a) with prompts as clusters.

    ``a`` and ``b`` have shape ``(n_prompts, n_seeds)`` and are aligned on both
    axes. NaNs (e.g. a pose metric undefined for one grid) are ignored inside a
    prompt; prompts that are NaN for every seed are dropped from both arms.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Expected two aligned 2-D arrays, got {a.shape} and {b.shape}")
    keep = ~(np.all(np.isnan(a), axis=1) | np.all(np.isnan(b), axis=1))
    a, b = a[keep], b[keep]
    n_prompts, n_seeds = a.shape
    if n_prompts < 2:
        raise ValueError("Need at least two prompts with defined values")

    rng = np.random.default_rng(seed)
    prompt_idx = rng.integers(0, n_prompts, size=(n_boot, n_prompts))
    seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_prompts, n_seeds))
    rows = prompt_idx[:, :, None]
    with np.errstate(all="ignore"):
        boot_a = np.nanmean(np.nanmean(a[rows, seed_idx], axis=2), axis=1)
        boot_b = np.nanmean(np.nanmean(b[rows, seed_idx], axis=2), axis=1)
    diffs = boot_b - boot_a
    diffs = diffs[~np.isnan(diffs)]
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(diffs, [tail, 1.0 - tail])

    mean_a = float(np.mean(_prompt_means(a)))
    mean_b = float(np.mean(_prompt_means(b)))
    return PairedEstimate(
        mean_a=mean_a,
        mean_b=mean_b,
        diff=mean_b - mean_a,
        ci_low=float(low),
        ci_high=float(high),
        n_prompts=n_prompts,
        n_seeds=n_seeds,
        n_boot=n_boot,
    )


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon p-value on per-prompt mean differences (needs scipy)."""
    from scipy.stats import wilcoxon

    diffs = _prompt_means(np.asarray(b, float)) - _prompt_means(np.asarray(a, float))
    diffs = diffs[~np.isnan(diffs)]
    if np.allclose(diffs, 0):
        return 1.0
    return float(wilcoxon(diffs, zero_method="pratt").pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def minimum_detectable_effect(
    null_prompt_diffs: np.ndarray,
    *,
    n_prompts: int,
    seeds_per_condition: int,
    seeds_in_null: int = 1,
) -> float:
    """Approximate paired MDE at alpha=0.05 (two-sided) and 80% power.

    ``null_prompt_diffs`` are per-prompt differences between two *identical*
    conditions (for example base seed 1001 minus base seed 1002), each averaged
    over ``seeds_in_null`` seeds. Their spread is pure generation noise. Averaging
    over ``seeds_per_condition`` seeds shrinks that noise by the square root of the
    seed ratio. This ignores any prompt-by-LoRA interaction, so it is optimistic:
    treat it as a lower bound on what the study can detect.
    """
    diffs = np.asarray(null_prompt_diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    sd = float(np.std(diffs, ddof=1))
    sd_scaled = sd * math.sqrt(seeds_in_null / seeds_per_condition)
    return (Z_975 + Z_80) * sd_scaled / math.sqrt(n_prompts)
