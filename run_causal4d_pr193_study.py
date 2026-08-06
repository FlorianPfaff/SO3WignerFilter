#!/usr/bin/env python3
"""Run claim-bounded scientific diagnostics for Causal4D PR #193.

The controlled study is executed from an immutable checkout of
IPS-Stuttgart/Causal4D commit
fa6a64b2442474321e453e9e8fdccd591e0a282d.  The design audit uses no target
outcomes and cannot modify the registered method or threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import numpy as np
from scipy.stats import nct, t

TARGET_SHA = "fa6a64b2442474321e453e9e8fdccd591e0a282d"
PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
RNG_SEED = 20_260_806
CALIBRATION_UNITS = 9
CALIBRATION_COVERAGE = 0.90
SIMULATION_DRAWS = 200_000


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def two_sided_t_power(
    independent_sessions: int,
    standardized_session_effect: float,
    *,
    alpha: float = 0.05,
) -> float:
    degrees = independent_sessions - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees))
    noncentrality = standardized_session_effect * math.sqrt(independent_sessions)
    return float(
        nct.cdf(-critical, degrees, noncentrality)
        + 1.0
        - nct.cdf(critical, degrees, noncentrality)
    )


def minimum_detectable_effect(
    independent_sessions: int,
    target_power: float,
) -> float:
    lower, upper = 0.0, 4.0
    for _ in range(100):
        middle = (lower + upper) / 2.0
        if two_sided_t_power(independent_sessions, middle) >= target_power:
            upper = middle
        else:
            lower = middle
    return upper


def quantiles(values: np.ndarray, levels: tuple[float, ...]) -> dict[str, float]:
    labels = tuple(f"q{int(round(level * 100)):02d}" for level in levels)
    return dict(zip(labels, map(float, np.quantile(values, levels)), strict=True))


def calibration_family(
    name: str,
    factory: Callable[[], np.ndarray],
) -> dict[str, Any]:
    scores = factory()
    ordered = np.sort(scores, axis=1)
    maximum = ordered[:, -1]
    second = ordered[:, -2]
    sample_median = np.median(scores, axis=1)
    population_median = max(float(np.median(scores)), 1e-15)
    maximum_to_sample_median = maximum / np.maximum(sample_median, 1e-15)
    second_to_maximum = second / np.maximum(maximum, 1e-15)
    maximum_to_population_median = maximum / population_median
    result = {
        "score_distribution": name,
        "simulation_draws": SIMULATION_DRAWS,
        "calibration_units": CALIBRATION_UNITS,
        "registered_rank_one_based": 9,
        "threshold_is_sample_maximum": True,
        "maximum_to_population_median_quantiles": quantiles(
            maximum_to_population_median,
            (0.05, 0.50, 0.90, 0.95, 0.99),
        ),
        "maximum_to_sample_median_quantiles": quantiles(
            maximum_to_sample_median,
            (0.50, 0.90, 0.95, 0.99),
        ),
        "probability_maximum_exceeds_twice_sample_median": float(
            np.mean(maximum_to_sample_median > 2.0)
        ),
        "probability_maximum_exceeds_three_times_sample_median": float(
            np.mean(maximum_to_sample_median > 3.0)
        ),
        "second_largest_to_maximum_quantiles": quantiles(
            second_to_maximum,
            (0.05, 0.50, 0.95),
        ),
        "fragility_may_change_registered_threshold": False,
    }
    del scores, ordered, maximum, second, sample_median
    return result


def build_design_audit() -> dict[str, Any]:
    effect_grid = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    endpoint_power = [
        {
            "endpoint": endpoint,
            "independent_sessions": sessions,
            "standardized_session_effect": effect,
            "two_sided_alpha": 0.05,
            "approximate_power": two_sided_t_power(sessions, effect),
        }
        for endpoint, sessions in (
            ("factual_continuation", 18),
            ("same_grasp_transfer", 18),
            ("new_contact_transfer", 12),
        )
        for effect in effect_grid
    ]

    precision = [
        {
            "independent_sessions": sessions,
            "expected_95_percent_half_width_in_session_sd_units": float(
                t.ppf(0.975, sessions - 1) / math.sqrt(sessions)
            ),
            "minimum_standardized_effect_for_80_percent_power": (
                minimum_detectable_effect(sessions, 0.80)
            ),
            "minimum_standardized_effect_for_90_percent_power": (
                minimum_detectable_effect(sessions, 0.90)
            ),
        }
        for sessions in (12, 18)
    ]

    factual_execution_sensitivity = []
    for within_session_correlation in (0.0, 0.25, 0.5, 0.75, 0.9):
        session_mean_sd_ratio = math.sqrt(
            (1.0 + within_session_correlation) / 2.0
        )
        for execution_effect in effect_grid:
            session_effect = execution_effect / session_mean_sd_ratio
            factual_execution_sensitivity.append(
                {
                    "within_session_execution_correlation": (
                        within_session_correlation
                    ),
                    "standardized_execution_effect": execution_effect,
                    "induced_standardized_session_mean_effect": session_effect,
                    "approximate_power_with_18_sessions": two_sided_t_power(
                        18, session_effect
                    ),
                }
            )

    registered_rank = math.ceil(
        (CALIBRATION_UNITS + 1) * CALIBRATION_COVERAGE
    )
    leave_one_out_units = CALIBRATION_UNITS - 1
    leave_one_out_rank = math.ceil(
        (leave_one_out_units + 1) * CALIBRATION_COVERAGE
    )

    rng = np.random.default_rng(RNG_SEED)
    factories: tuple[tuple[str, Callable[[], np.ndarray]], ...] = (
        ("half_normal", lambda: np.abs(rng.normal(size=(SIMULATION_DRAWS, 9)))),
        (
            "lognormal_sigma_0.25",
            lambda: rng.lognormal(0.0, 0.25, size=(SIMULATION_DRAWS, 9)),
        ),
        (
            "lognormal_sigma_0.50",
            lambda: rng.lognormal(0.0, 0.50, size=(SIMULATION_DRAWS, 9)),
        ),
        (
            "lognormal_sigma_1.00",
            lambda: rng.lognormal(0.0, 1.00, size=(SIMULATION_DRAWS, 9)),
        ),
        (
            "absolute_t5",
            lambda: np.abs(rng.standard_t(df=5, size=(SIMULATION_DRAWS, 9))),
        ),
    )
    calibration_fragility = [
        calibration_family(name, factory) for name, factory in factories
    ]

    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DRegisteredDesignPowerAndFragilityAudit",
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_DESIGN_SHA256,
        "session_cluster_is_independent_unit": True,
        "endpoint_power": endpoint_power,
        "precision_and_minimum_detectable_effect": precision,
        "factual_two_execution_session_sensitivity": factual_execution_sensitivity,
        "execution_block_calibration": {
            "coverage_target": CALIBRATION_COVERAGE,
            "calibration_units": CALIBRATION_UNITS,
            "registered_rank_one_based": registered_rank,
            "registered_rank_is_sample_maximum": (
                registered_rank == CALIBRATION_UNITS
            ),
            "exchangeable_finite_sample_coverage": (
                registered_rank / (CALIBRATION_UNITS + 1)
            ),
            "leave_one_session_out_units": leave_one_out_units,
            "leave_one_session_out_rank_one_based": leave_one_out_rank,
            "leave_one_session_out_threshold_is_finite": (
                leave_one_out_rank <= leave_one_out_units
            ),
            "registered_infinite_sentinel_required_after_one_deletion": (
                leave_one_out_rank > leave_one_out_units
            ),
            "simulation_rng_seed": RNG_SEED,
            "simulation_draws_per_score_family": SIMULATION_DRAWS,
            "score_family_count": len(calibration_fragility),
            "fragility_scenarios": calibration_fragility,
        },
        "interpretation": {
            "not_conditioned_on_target_outcomes": True,
            "does_not_change_registered_method_or_threshold": True,
            "does_not_rescue_a_failed_registered_endpoint": True,
            "does_not_substitute_for_the_36_execution_physical_result": True,
        },
    }


def write_design_markdown(path: Path, audit: dict[str, Any]) -> None:
    precision = audit["precision_and_minimum_detectable_effect"]
    lines = [
        "# Causal4D PR #193 registered-design audit",
        "",
        "This is a source-only diagnostic. It uses no target outcomes and cannot alter the registered method, threshold, exclusions, or decision rules.",
        "",
        "## Precision and minimum detectable standardized session effects",
        "",
        "| Independent sessions | 95% half-width (session SD) | d for 80% power | d for 90% power |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in precision:
        lines.append(
            "| {independent_sessions} | "
            "{expected_95_percent_half_width_in_session_sd_units:.3f} | "
            "{minimum_standardized_effect_for_80_percent_power:.3f} | "
            "{minimum_standardized_effect_for_90_percent_power:.3f} |".format(**row)
        )
    calibration = audit["execution_block_calibration"]
    lines.extend(
        [
            "",
            "## Execution-block calibration boundary",
            "",
            f"With {calibration['calibration_units']} calibration sessions, the registered 90% split-conformal rank is {calibration['registered_rank_one_based']} of {calibration['calibration_units']}: the sample maximum.",
            f"After deleting one calibration session, the formal rank is {calibration['leave_one_session_out_rank_one_based']} with only {calibration['leave_one_session_out_units']} available scores, so the registered infinite sentinel is required.",
            "",
            "These facts characterize utility and fragility only; they do not authorize threshold selection after target access.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--controlled-profile",
        choices=("smoke", "standard", "full"),
        default="standard",
    )
    arguments = parser.parse_args()

    target_root = arguments.target_root.resolve()
    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    actual_sha = git_output(target_root, "rev-parse", "HEAD")
    if actual_sha != TARGET_SHA:
        raise SystemExit(f"immutable target mismatch: {actual_sha} != {TARGET_SHA}")
    if git_output(target_root, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("immutable target checkout is dirty")

    design_audit = build_design_audit()
    design_path = output_root / "registered-design-power-fragility.json"
    write_json(design_path, design_audit)
    write_design_markdown(output_root / "registered-design-power-fragility.md", design_audit)

    controlled_root = output_root / f"controlled-{arguments.controlled_profile}"
    controlled_log = output_root / f"controlled-{arguments.controlled_profile}.log"
    command = [
        sys.executable,
        str(target_root / "scripts/ci/run_self_hosted_evaluation.py"),
        "--profile",
        arguments.controlled_profile,
        "--output-dir",
        str(controlled_root),
    ]
    with controlled_log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    controlled_summary_path = controlled_root / "summary.json"
    controlled_summary = load_json(controlled_summary_path)
    summary = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPR193ScientificStudySummary",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "target_repository": "IPS-Stuttgart/Causal4D",
        "target_sha": actual_sha,
        "protocol_id": PROTOCOL_ID,
        "protocol_design_sha256": PROTOCOL_DESIGN_SHA256,
        "design_audit_sha256": sha256_file(design_path),
        "controlled_profile": arguments.controlled_profile,
        "controlled_returncode": int(completed.returncode),
        "controlled_completed": controlled_summary is not None,
        "controlled_integrity": (
            controlled_summary.get("integrity")
            if controlled_summary is not None
            else None
        ),
        "controlled_summary_sha256": (
            sha256_file(controlled_summary_path)
            if controlled_summary_path.is_file()
            else None
        ),
        "precision_and_minimum_detectable_effect": design_audit[
            "precision_and_minimum_detectable_effect"
        ],
        "calibration_boundary": design_audit["execution_block_calibration"],
        "physical_evidence_evaluated": False,
        "claim_boundary": (
            "The design audit and controlled simulation are not substitutes for "
            "the registered 18-session/36-execution physical result. A positive "
            "physical claim remains bounded to sloth_plush_instance_1 and the "
            "frozen action/contact/hardware configuration."
        ),
    }
    write_json(output_root / "study-summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
