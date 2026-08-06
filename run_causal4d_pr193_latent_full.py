#!/usr/bin/env python3
"""Run the predeclared full-scale Causal4D latent-contact benchmark.

The study evaluates the immutable implementation with the exact `full` profile
latent seed range (2000..2049). It does not tune gates or inspect physical
experiment outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark,
    write_latent_contact_artifacts,
)
from causal4d.contact_inference import LatentContactConfig

TARGET_SHA = "fa6a64b2442474321e453e9e8fdccd591e0a282d"
SEEDS = tuple(range(2_000, 2_050))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [math.nan, math.nan]
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def read_contact_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().casefold() == "true"


def summarize_accuracy(rows: list[dict[str, str]]) -> dict[str, Any]:
    shifted_online = [
        row
        for row in rows
        if row["world_condition"] == "shifted_contact"
        and row["setting"] == "online_adaptation"
    ]
    successes = sum(as_bool(row["node_correct"]) for row in shifted_online)
    topology = {}
    for object_id in sorted({row["object"] for row in shifted_online}):
        selected = [row for row in shifted_online if row["object"] == object_id]
        correct = sum(as_bool(row["node_correct"]) for row in selected)
        topology[object_id] = {
            "successes": correct,
            "trials": len(selected),
            "accuracy": correct / len(selected),
            "wilson_ci95": wilson_interval(correct, len(selected)),
        }
    return {
        "successes": successes,
        "trials": len(shifted_online),
        "accuracy": successes / len(shifted_online),
        "wilson_ci95": wilson_interval(successes, len(shifted_online)),
        "frozen_gate": 0.80,
        "passed": successes / len(shifted_online) >= 0.80,
        "by_object": topology,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    benchmark_config = CounterfactualBenchmarkConfig()
    contact_config = LatentContactConfig(
        parameter_particle_count=12,
        observation_fraction=0.20,
        observation_noise_std_m=0.0015,
        confidence_level=benchmark_config.confidence_level,
    )
    started = time.perf_counter()
    result = run_latent_contact_benchmark(
        seeds=list(SEEDS),
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    elapsed = time.perf_counter() - started
    artifact_paths = write_latent_contact_artifacts(result, output / "artifacts")
    contact_rows = read_contact_rows(output / "artifacts" / "contact_recovery.csv")
    node_accuracy = summarize_accuracy(contact_rows)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPR193FullLatentContactStudy",
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "target_sha": TARGET_SHA,
        "profile": "full-latent-only",
        "seed_first": SEEDS[0],
        "seed_last": SEEDS[-1],
        "seed_count": len(SEEDS),
        "elapsed_seconds": elapsed,
        "benchmark_config": benchmark_config.as_dict(),
        "contact_config": contact_config.as_dict(),
        "success_gates": result["success_gates"],
        "aggregate": result["aggregate"],
        "shifted_online_node_accuracy": node_accuracy,
        "artifact_paths": artifact_paths,
        "gate_thresholds_unchanged": True,
        "uses_physical_target_outcomes": False,
        "claim_boundary": (
            "Controlled latent-contact diagnostic only; not a substitute for the "
            "registered 18-session/36-execution physical experiment."
        ),
    }
    summary["study_id"] = canonical_digest(summary)
    write_json(output / "summary.json", summary)

    lines = [
        "# Causal4D PR #193 full latent-contact study",
        "",
        f"- Seeds: `{SEEDS[0]}..{SEEDS[-1]}` ({len(SEEDS)} total)",
        f"- Overall frozen gate: `{result['success_gates'].get('overall_passed')}`",
        f"- Shifted online node accuracy: `{node_accuracy['successes']}/{node_accuracy['trials']} = {node_accuracy['accuracy']:.4f}`",
        f"- Wilson 95% interval: `[{node_accuracy['wilson_ci95'][0]:.4f}, {node_accuracy['wilson_ci95'][1]:.4f}]`",
        "",
        "No gate or threshold was tuned after observing this result.",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
