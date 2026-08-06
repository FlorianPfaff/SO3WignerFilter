#!/usr/bin/env python3
"""Audit the exact frozen Causal4D session-bootstrap operating characteristics.

This is a source-only simulation. It imports the immutable target's bootstrap
constants and verifies a matrix implementation against the target function
before estimating small-sample coverage and detection power. It does not use
physical target outcomes or alter the registered method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

from causal4d import real_analysis_reporting as reporting

TARGET_SHA = "fa6a64b2442474321e453e9e8fdccd591e0a282d"
PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
OUTER_REPLICATES = 2_000
OUTER_SEED = 20_260_806
BATCH_SIZE = 100
EFFECT_GRID = (0.0, 0.35, 0.50, 0.70, 0.90, 1.00)

FloatArray = NDArray[np.float64]
Generator = Callable[[np.random.Generator, int, int], FloatArray]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def normal_generator(
    rng: np.random.Generator,
    batch_size: int,
    sample_count: int,
) -> FloatArray:
    return np.asarray(rng.normal(size=(batch_size, sample_count)), dtype=np.float64)


def t5_generator(
    rng: np.random.Generator,
    batch_size: int,
    sample_count: int,
) -> FloatArray:
    # Student-t variance is nu / (nu - 2), so scale to population variance one.
    scale = math.sqrt(3.0 / 5.0)
    return np.asarray(
        scale * rng.standard_t(df=5, size=(batch_size, sample_count)),
        dtype=np.float64,
    )


def lognormal_generator(
    sigma: float,
) -> Generator:
    population_mean = math.exp(0.5 * sigma * sigma)
    population_variance = (
        math.exp(sigma * sigma) - 1.0
    ) * math.exp(sigma * sigma)
    population_sd = math.sqrt(population_variance)

    def generate(
        rng: np.random.Generator,
        batch_size: int,
        sample_count: int,
    ) -> FloatArray:
        raw = rng.lognormal(
            mean=0.0,
            sigma=sigma,
            size=(batch_size, sample_count),
        )
        return np.asarray(
            (raw - population_mean) / population_sd,
            dtype=np.float64,
        )

    return generate


def contaminated_normal_generator(
    rng: np.random.Generator,
    batch_size: int,
    sample_count: int,
) -> FloatArray:
    outlier = rng.random(size=(batch_size, sample_count)) < 0.05
    scales = np.where(outlier, 5.0, 1.0)
    raw = rng.normal(size=(batch_size, sample_count)) * scales
    population_sd = math.sqrt(0.95 + 0.05 * 25.0)
    return np.asarray(raw / population_sd, dtype=np.float64)


def quantile_summary(values: FloatArray) -> dict[str, float]:
    levels = (0.05, 0.25, 0.50, 0.75, 0.95)
    labels = ("q05", "q25", "q50", "q75", "q95")
    return dict(
        zip(labels, map(float, np.quantile(values, levels)), strict=True)
    )


def audit_scenario(
    *,
    name: str,
    generator: Generator,
    sample_count: int,
    weights: FloatArray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lower_parts: list[FloatArray] = []
    upper_parts: list[FloatArray] = []
    sample_mean_parts: list[FloatArray] = []
    t_lower_parts: list[FloatArray] = []
    t_upper_parts: list[FloatArray] = []
    remaining = OUTER_REPLICATES
    verified_target_function = False
    verification_error = 0.0
    tail = 0.5 * (1.0 - reporting.BOOTSTRAP_CONFIDENCE_LEVEL)
    t_critical = float(
        t.ppf(
            0.5 * (1.0 + reporting.BOOTSTRAP_CONFIDENCE_LEVEL),
            sample_count - 1,
        )
    )

    while remaining:
        current = min(BATCH_SIZE, remaining)
        samples = generator(rng, current, sample_count)
        sample_means = np.mean(samples, axis=1)
        bootstrap_means = samples @ weights.T
        lower, upper = np.quantile(
            bootstrap_means,
            [tail, 1.0 - tail],
            axis=1,
        )
        sample_sd = np.std(samples, axis=1, ddof=1)
        t_half_width = t_critical * sample_sd / math.sqrt(sample_count)
        t_lower = sample_means - t_half_width
        t_upper = sample_means + t_half_width

        if not verified_target_function:
            target = reporting._bootstrap(samples[0].tolist())
            verification_error = max(
                abs(float(target["lower"]) - float(lower[0])),
                abs(float(target["upper"]) - float(upper[0])),
            )
            if verification_error > 1e-12:
                raise RuntimeError(
                    "matrix bootstrap differs from immutable target function: "
                    f"error={verification_error}"
                )
            verified_target_function = True

        lower_parts.append(np.asarray(lower, dtype=np.float64))
        upper_parts.append(np.asarray(upper, dtype=np.float64))
        sample_mean_parts.append(np.asarray(sample_means, dtype=np.float64))
        t_lower_parts.append(np.asarray(t_lower, dtype=np.float64))
        t_upper_parts.append(np.asarray(t_upper, dtype=np.float64))
        remaining -= current

    lower = np.concatenate(lower_parts)
    upper = np.concatenate(upper_parts)
    sample_mean = np.concatenate(sample_mean_parts)
    t_lower = np.concatenate(t_lower_parts)
    t_upper = np.concatenate(t_upper_parts)
    width = upper - lower
    left_half = sample_mean - lower
    right_half = upper - sample_mean
    bootstrap_covered = (lower <= 0.0) & (upper >= 0.0)
    t_covered = (t_lower <= 0.0) & (t_upper >= 0.0)
    coverage = float(np.mean(bootstrap_covered))
    coverage_mcse = math.sqrt(coverage * (1.0 - coverage) / OUTER_REPLICATES)

    power = []
    for effect in EFFECT_GRID:
        power.append(
            {
                "standardized_population_mean_effect": effect,
                "bootstrap_favorable_detection_probability": float(
                    np.mean(lower + effect > 0.0)
                ),
                "bootstrap_adverse_detection_probability": float(
                    np.mean(upper + effect < 0.0)
                ),
                "t_interval_favorable_detection_probability": float(
                    np.mean(t_lower + effect > 0.0)
                ),
            }
        )

    return {
        "distribution": name,
        "sample_count": sample_count,
        "population_mean": 0.0,
        "population_sd": 1.0,
        "outer_replicates": OUTER_REPLICATES,
        "outer_seed": seed,
        "bootstrap_replicates": reporting.BOOTSTRAP_REPLICATES,
        "bootstrap_seed": reporting.BOOTSTRAP_SEED,
        "bootstrap_confidence_level": reporting.BOOTSTRAP_CONFIDENCE_LEVEL,
        "exact_target_function_verified": verified_target_function,
        "maximum_target_function_verification_error": verification_error,
        "bootstrap_null_coverage": coverage,
        "bootstrap_null_coverage_mcse": coverage_mcse,
        "bootstrap_favorable_type_i_error": float(np.mean(lower > 0.0)),
        "bootstrap_adverse_type_i_error": float(np.mean(upper < 0.0)),
        "t_interval_null_coverage": float(np.mean(t_covered)),
        "interval_width_quantiles": quantile_summary(width),
        "left_half_width_quantiles": quantile_summary(left_half),
        "right_half_width_quantiles": quantile_summary(right_half),
        "sample_mean_quantiles": quantile_summary(sample_mean),
        "detection_power": power,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Causal4D PR #193 exact session-bootstrap audit",
        "",
        "The immutable target's 20,000-replicate percentile bootstrap and fixed seed were used exactly. No target outcomes were read.",
        "",
        "| n | Distribution | Bootstrap coverage | MCSE | Favorable type-I | t coverage | Median width |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in payload["scenarios"]:
        lines.append(
            "| {sample_count} | {distribution} | {bootstrap_null_coverage:.3f} | "
            "{bootstrap_null_coverage_mcse:.3f} | "
            "{bootstrap_favorable_type_i_error:.3f} | "
            "{t_interval_null_coverage:.3f} | {width:.3f} |".format(
                width=scenario["interval_width_quantiles"]["q50"],
                **scenario,
            )
        )
    lines.extend(
        [
            "",
            "Coverage and power are operating-characteristic diagnostics only. They cannot change the frozen bootstrap or rescue a failed physical endpoint.",
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
        raise SystemExit("immutable target bootstrap confidence level changed")

    generators: tuple[tuple[str, Generator], ...] = (
        ("normal", normal_generator),
        ("student_t5_variance_one", t5_generator),
        ("centered_lognormal_sigma_0.50", lognormal_generator(0.50)),
        ("centered_lognormal_sigma_1.00", lognormal_generator(1.00)),
        ("five_percent_scale5_contaminated_normal", contaminated_normal_generator),
    )
    scenarios = []
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
        "artifact_kind": "Causal4DSessionBootstrapOperatingCharacteristics",
        "target_sha": TARGET_SHA,
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_DESIGN_SHA256,
        "target_bootstrap": {
            "method": "session_cluster_percentile_bootstrap",
            "replicates": reporting.BOOTSTRAP_REPLICATES,
            "seed": reporting.BOOTSTRAP_SEED,
            "confidence_level": reporting.BOOTSTRAP_CONFIDENCE_LEVEL,
        },
        "outer_replicates_per_scenario": OUTER_REPLICATES,
        "scenario_count": len(scenarios),
        "total_synthetic_session_panels": OUTER_REPLICATES * len(scenarios),
        "effect_grid": list(EFFECT_GRID),
        "scenarios": scenarios,
        "uses_target_outcomes": False,
        "may_change_registered_method": False,
        "may_change_primary_decision": False,
        "claim_boundary": (
            "Source-only operating-characteristic audit; not a substitute for "
            "the registered 18-session/36-execution physical experiment."
        ),
    }
    payload["audit_id"] = canonical_digest(payload)
    write_json(output / "session-bootstrap-operating-characteristics.json", payload)
    write_markdown(output / "session-bootstrap-operating-characteristics.md", payload)

    compact = {
        "schema_version": 1,
        "artifact_kind": "Causal4DSessionBootstrapAuditSummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "target_sha": TARGET_SHA,
        "audit_id": payload["audit_id"],
        "scenario_count": len(scenarios),
        "total_synthetic_session_panels": payload[
            "total_synthetic_session_panels"
        ],
        "all_target_function_checks_passed": all(
            scenario["exact_target_function_verified"] for scenario in scenarios
        ),
        "coverage_by_scenario": [
            {
                "sample_count": scenario["sample_count"],
                "distribution": scenario["distribution"],
                "bootstrap_null_coverage": scenario[
                    "bootstrap_null_coverage"
                ],
                "bootstrap_null_coverage_mcse": scenario[
                    "bootstrap_null_coverage_mcse"
                ],
                "bootstrap_favorable_type_i_error": scenario[
                    "bootstrap_favorable_type_i_error"
                ],
                "t_interval_null_coverage": scenario[
                    "t_interval_null_coverage"
                ],
            }
            for scenario in scenarios
        ],
        "uses_target_outcomes": False,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(output / "summary.json", compact)
    print(json.dumps(compact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
