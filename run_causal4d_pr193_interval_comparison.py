#!/usr/bin/env python3
"""Compare small-sample interval methods for Causal4D PR #193.

The comparison is source-only and preacquisition. It uses the immutable
Causal4D bootstrap constants and identical synthetic session panels to compare
current percentile, basic, Student-t, BCa, and bootstrap-t intervals for the
mean session effect. No physical target outcome is read and no method is
selected or changed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm, t

from causal4d import real_analysis_reporting as reporting

from run_causal4d_pr193_bootstrap_audit import (
    EFFECT_GRID,
    Generator,
    TARGET_SHA,
    contaminated_normal_generator,
    lognormal_generator,
    normal_generator,
    t5_generator,
    write_json,
)

PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
OUTER_REPLICATES = 1_500
OUTER_SEED = 20_260_807
BATCH_SIZE = 50
METHODS = ("percentile", "basic", "student_t", "bca", "bootstrap_t")

FloatArray = NDArray[np.float64]


def canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def bootstrap_weights(sample_count: int) -> FloatArray:
    rng = np.random.default_rng(reporting.BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        sample_count,
        size=(reporting.BOOTSTRAP_REPLICATES, sample_count),
    )
    counts = np.zeros(
        (reporting.BOOTSTRAP_REPLICATES, sample_count),
        dtype=np.float64,
    )
    rows = np.repeat(np.arange(reporting.BOOTSTRAP_REPLICATES), sample_count)
    np.add.at(counts, (rows, indices.ravel()), 1.0)
    return counts / float(sample_count)


def rowwise_quantile_from_sorted(
    sorted_values: FloatArray,
    probabilities: FloatArray,
) -> FloatArray:
    """NumPy-compatible linear quantiles with one probability per row."""

    count = sorted_values.shape[1]
    positions = np.clip(probabilities, 0.0, 1.0) * float(count - 1)
    lower_index = np.floor(positions).astype(np.int64)
    upper_index = np.ceil(positions).astype(np.int64)
    fraction = positions - lower_index
    rows = np.arange(sorted_values.shape[0])
    lower = sorted_values[rows, lower_index]
    upper = sorted_values[rows, upper_index]
    return lower + fraction * (upper - lower)


def fixed_quantile_from_sorted(
    sorted_values: FloatArray,
    probability: float,
) -> FloatArray:
    probabilities = np.full(sorted_values.shape[0], probability, dtype=np.float64)
    return rowwise_quantile_from_sorted(sorted_values, probabilities)


def bca_bounds(
    *,
    samples: FloatArray,
    sorted_bootstrap_means: FloatArray,
    sample_means: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    bootstrap_count = sorted_bootstrap_means.shape[1]
    less_fraction = np.mean(
        sorted_bootstrap_means < sample_means[:, None],
        axis=1,
    )
    epsilon = 0.5 / float(bootstrap_count)
    z0 = norm.ppf(np.clip(less_fraction, epsilon, 1.0 - epsilon))

    sample_count = samples.shape[1]
    jackknife = (
        sample_count * sample_means[:, None] - samples
    ) / float(sample_count - 1)
    jackknife_mean = np.mean(jackknife, axis=1, keepdims=True)
    centered = jackknife_mean - jackknife
    numerator = np.sum(centered**3, axis=1)
    denominator = 6.0 * np.power(np.sum(centered**2, axis=1), 1.5)
    acceleration = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )

    tail = 0.5 * (1.0 - reporting.BOOTSTRAP_CONFIDENCE_LEVEL)
    adjusted = []
    for probability in (tail, 1.0 - tail):
        z_alpha = float(norm.ppf(probability))
        numerator_z = z0 + z_alpha
        denominator_z = 1.0 - acceleration * numerator_z
        transformed = z0 + np.divide(
            numerator_z,
            denominator_z,
            out=np.sign(numerator_z) * np.full_like(numerator_z, np.inf),
            where=np.abs(denominator_z) > 1e-15,
        )
        adjusted.append(np.clip(norm.cdf(transformed), 0.0, 1.0))
    return (
        rowwise_quantile_from_sorted(sorted_bootstrap_means, adjusted[0]),
        rowwise_quantile_from_sorted(sorted_bootstrap_means, adjusted[1]),
    )


def bootstrap_t_bounds(
    *,
    samples: FloatArray,
    weights: FloatArray,
    bootstrap_means: FloatArray,
    sample_means: FloatArray,
    sample_sd: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    sample_count = samples.shape[1]
    bootstrap_second_moment = (samples * samples) @ weights.T
    bootstrap_variance = (
        float(sample_count)
        / float(sample_count - 1)
        * np.maximum(
            bootstrap_second_moment - bootstrap_means * bootstrap_means,
            0.0,
        )
    )
    bootstrap_se = np.sqrt(bootstrap_variance / float(sample_count))
    sample_se = sample_sd / math.sqrt(sample_count)
    t_star = np.divide(
        bootstrap_means - sample_means[:, None],
        bootstrap_se,
        out=np.full_like(bootstrap_means, np.nan),
        where=bootstrap_se > 0.0,
    )
    # Degenerate resamples are rare for continuous inputs. Replace them with
    # signed infinities so quantiles remain conservative rather than silently
    # dropping a source-defined resample.
    invalid = ~np.isfinite(t_star)
    if np.any(invalid):
        signed = np.sign(bootstrap_means - sample_means[:, None])
        t_star[invalid] = np.where(signed[invalid] >= 0.0, np.inf, -np.inf)
    lower_t, upper_t = np.quantile(
        t_star,
        [
            0.5 * (1.0 - reporting.BOOTSTRAP_CONFIDENCE_LEVEL),
            1.0 - 0.5 * (1.0 - reporting.BOOTSTRAP_CONFIDENCE_LEVEL),
        ],
        axis=1,
    )
    return (
        sample_means - upper_t * sample_se,
        sample_means - lower_t * sample_se,
    )


def summarize_method(
    *,
    lower: FloatArray,
    upper: FloatArray,
) -> dict[str, Any]:
    covered = (lower <= 0.0) & (upper >= 0.0)
    coverage = float(np.mean(covered))
    return {
        "null_coverage": coverage,
        "null_coverage_mcse": math.sqrt(
            coverage * (1.0 - coverage) / OUTER_REPLICATES
        ),
        "favorable_type_i_error": float(np.mean(lower > 0.0)),
        "adverse_type_i_error": float(np.mean(upper < 0.0)),
        "median_interval_width": float(np.median(upper - lower)),
        "mean_interval_width": float(np.mean(upper - lower)),
        "detection_power": [
            {
                "standardized_population_mean_effect": effect,
                "favorable_detection_probability": float(
                    np.mean(lower + effect > 0.0)
                ),
            }
            for effect in EFFECT_GRID
        ],
    }


def audit_scenario(
    *,
    name: str,
    generator: Generator,
    sample_count: int,
    weights: FloatArray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    bounds: dict[str, list[tuple[FloatArray, FloatArray]]] = {
        method: [] for method in METHODS
    }
    remaining = OUTER_REPLICATES
    target_verification_error = 0.0
    target_verified = False
    tail = 0.5 * (1.0 - reporting.BOOTSTRAP_CONFIDENCE_LEVEL)
    t_critical = float(
        t.ppf(
            1.0 - tail,
            sample_count - 1,
        )
    )

    while remaining:
        current = min(BATCH_SIZE, remaining)
        samples = generator(rng, current, sample_count)
        sample_means = np.mean(samples, axis=1)
        sample_sd = np.std(samples, axis=1, ddof=1)
        bootstrap_means = samples @ weights.T
        sorted_bootstrap_means = np.sort(bootstrap_means, axis=1)
        percentile_lower = fixed_quantile_from_sorted(
            sorted_bootstrap_means, tail
        )
        percentile_upper = fixed_quantile_from_sorted(
            sorted_bootstrap_means, 1.0 - tail
        )
        basic_lower = 2.0 * sample_means - percentile_upper
        basic_upper = 2.0 * sample_means - percentile_lower
        half_width = t_critical * sample_sd / math.sqrt(sample_count)
        student_lower = sample_means - half_width
        student_upper = sample_means + half_width
        bca_lower, bca_upper = bca_bounds(
            samples=samples,
            sorted_bootstrap_means=sorted_bootstrap_means,
            sample_means=sample_means,
        )
        boot_t_lower, boot_t_upper = bootstrap_t_bounds(
            samples=samples,
            weights=weights,
            bootstrap_means=bootstrap_means,
            sample_means=sample_means,
            sample_sd=sample_sd,
        )

        if not target_verified:
            target = reporting._bootstrap(samples[0].tolist())
            target_verification_error = max(
                abs(float(target["lower"]) - float(percentile_lower[0])),
                abs(float(target["upper"]) - float(percentile_upper[0])),
            )
            if target_verification_error > 1e-12:
                raise RuntimeError(
                    "percentile implementation differs from immutable target: "
                    f"error={target_verification_error}"
                )
            target_verified = True

        for method, pair in (
            ("percentile", (percentile_lower, percentile_upper)),
            ("basic", (basic_lower, basic_upper)),
            ("student_t", (student_lower, student_upper)),
            ("bca", (bca_lower, bca_upper)),
            ("bootstrap_t", (boot_t_lower, boot_t_upper)),
        ):
            bounds[method].append(pair)
        remaining -= current

    summaries = {}
    for method, parts in bounds.items():
        lower = np.concatenate([part[0] for part in parts])
        upper = np.concatenate([part[1] for part in parts])
        summaries[method] = summarize_method(lower=lower, upper=upper)
    return {
        "distribution": name,
        "sample_count": sample_count,
        "population_mean": 0.0,
        "population_sd": 1.0,
        "outer_replicates": OUTER_REPLICATES,
        "outer_seed": seed,
        "target_percentile_verified": target_verified,
        "target_percentile_verification_error": target_verification_error,
        "methods": summaries,
    }


def aggregate_method(scenarios: list[dict[str, Any]], method: str) -> dict[str, Any]:
    coverage = np.asarray(
        [scenario["methods"][method]["null_coverage"] for scenario in scenarios],
        dtype=np.float64,
    )
    widths = np.asarray(
        [
            scenario["methods"][method]["median_interval_width"]
            for scenario in scenarios
        ],
        dtype=np.float64,
    )
    favorable = np.asarray(
        [
            scenario["methods"][method]["favorable_type_i_error"]
            for scenario in scenarios
        ],
        dtype=np.float64,
    )
    return {
        "method": method,
        "mean_absolute_coverage_error": float(np.mean(np.abs(coverage - 0.95))),
        "maximum_absolute_coverage_error": float(np.max(np.abs(coverage - 0.95))),
        "minimum_coverage": float(np.min(coverage)),
        "maximum_favorable_type_i_error": float(np.max(favorable)),
        "median_of_scenario_median_widths": float(np.median(widths)),
        "may_change_registered_method": False,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Causal4D PR #193 interval-method comparison",
        "",
        "All methods used identical synthetic session panels and the immutable target's fixed 20,000 bootstrap resamples. No target outcomes were read.",
        "",
        "| n | Distribution | Method | Coverage | Favorable type-I | Median width |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for scenario in payload["scenarios"]:
        for method in METHODS:
            result = scenario["methods"][method]
            lines.append(
                "| {sample_count} | {distribution} | {method} | {coverage:.3f} | "
                "{type_i:.3f} | {width:.3f} |".format(
                    sample_count=scenario["sample_count"],
                    distribution=scenario["distribution"],
                    method=method,
                    coverage=result["null_coverage"],
                    type_i=result["favorable_type_i_error"],
                    width=result["median_interval_width"],
                )
            )
    lines.extend(
        [
            "",
            "## Aggregate diagnostics",
            "",
            "| Method | Mean absolute coverage error | Worst absolute error | Minimum coverage | Maximum favorable type-I |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in payload["method_aggregates"]:
        lines.append(
            "| {method} | {mean_absolute_coverage_error:.3f} | "
            "{maximum_absolute_coverage_error:.3f} | {minimum_coverage:.3f} | "
            "{maximum_favorable_type_i_error:.3f} |".format(**result)
        )
    lines.extend(
        [
            "",
            "This audit does not automatically select a replacement. A protocol amendment, if justified, must remain explicit and preacquisition.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if reporting.BOOTSTRAP_REPLICATES != 20_000:
        raise SystemExit("immutable target bootstrap replicate count changed")
    if reporting.BOOTSTRAP_SEED != 20_260_726:
        raise SystemExit("immutable target bootstrap seed changed")
    if reporting.BOOTSTRAP_CONFIDENCE_LEVEL != 0.95:
        raise SystemExit("immutable target confidence level changed")

    generators: tuple[tuple[str, Generator], ...] = (
        ("normal", normal_generator),
        ("student_t5_variance_one", t5_generator),
        ("centered_lognormal_sigma_0.50", lognormal_generator(0.50)),
        ("centered_lognormal_sigma_1.00", lognormal_generator(1.00)),
        ("five_percent_scale5_contaminated_normal", contaminated_normal_generator),
    )
    scenarios: list[dict[str, Any]] = []
    for sample_count in (12, 18):
        weights = bootstrap_weights(sample_count)
        for distribution_index, (name, generator) in enumerate(generators):
            scenarios.append(
                audit_scenario(
                    name=name,
                    generator=generator,
                    sample_count=sample_count,
                    weights=weights,
                    seed=(
                        OUTER_SEED
                        + 10_000 * sample_count
                        + distribution_index
                    ),
                )
            )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPreacquisitionIntervalMethodComparison",
        "target_sha": TARGET_SHA,
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_DESIGN_SHA256,
        "target_bootstrap": {
            "replicates": reporting.BOOTSTRAP_REPLICATES,
            "seed": reporting.BOOTSTRAP_SEED,
            "confidence_level": reporting.BOOTSTRAP_CONFIDENCE_LEVEL,
        },
        "outer_replicates_per_scenario": OUTER_REPLICATES,
        "scenario_count": len(scenarios),
        "total_synthetic_session_panels": OUTER_REPLICATES * len(scenarios),
        "methods": list(METHODS),
        "effect_grid": list(EFFECT_GRID),
        "scenarios": scenarios,
        "method_aggregates": [
            aggregate_method(scenarios, method) for method in METHODS
        ],
        "all_target_percentile_checks_passed": all(
            scenario["target_percentile_verified"] for scenario in scenarios
        ),
        "uses_target_outcomes": False,
        "automatically_selects_method": False,
        "may_change_registered_method": False,
        "claim_boundary": (
            "Source-only preacquisition comparison; no replacement method is "
            "selected without an explicit protocol amendment."
        ),
    }
    payload["audit_id"] = canonical_digest(payload)
    write_json(output / "interval-method-comparison.json", payload)
    write_markdown(output / "interval-method-comparison.md", payload)
    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DIntervalMethodComparisonSummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "target_sha": TARGET_SHA,
        "audit_id": payload["audit_id"],
        "scenario_count": len(scenarios),
        "total_synthetic_session_panels": payload[
            "total_synthetic_session_panels"
        ],
        "method_aggregates": payload["method_aggregates"],
        "all_target_percentile_checks_passed": payload[
            "all_target_percentile_checks_passed"
        ],
        "uses_target_outcomes": False,
        "automatically_selects_method": False,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
