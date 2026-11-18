#!/usr/bin/env python3
"""Validate and analyze the locked matched-pool ensemble."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = ["fixed", "random", "feedback"]
POLICY_COLORS = {
    "fixed": "#4C6A92",
    "random": "#7A9A65",
    "feedback": "#B06A5B",
}
PANEL_LABEL_COLOR = "#24364B"
PRIMARY_METRIC = "shape_anisotropy"
METRICS = [
    "shape_anisotropy",
    "stiffness_anisotropy",
    "plastic_localization_cv",
    "plastic_strain_p95",
    "plastic_strain_p99",
    "plastified_volume_fraction",
    "com_magnitude",
    "volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--mesh-study-dir")
    parser.add_argument("--expected-pool-seeds", nargs="*", type=int)
    parser.add_argument("--expected-config")
    return parser.parse_args()


def load_complete_runs(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    quality_rows = []
    for metadata_path in sorted(input_dir.glob("*_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_id = metadata["run_id"]
        if metadata.get("status") != "complete":
            raise RuntimeError(
                f"Run {run_id} has status {metadata.get('status')!r}; "
                "analysis requires a completely finished input directory."
            )
        metrics_path = input_dir / f"{run_id}_cycle_metrics.csv"
        diagnostics_path = input_dir / f"{run_id}_solver_diagnostics.csv"
        pool_path = input_dir / f"{run_id}_load_pool.csv"
        if not metrics_path.exists() or not diagnostics_path.exists() or not pool_path.exists():
            raise RuntimeError(f"Complete run {run_id} is missing required output files.")
        metrics = pd.read_csv(metrics_path)
        diagnostics = pd.read_csv(diagnostics_path)
        pool = pd.read_csv(pool_path).sort_values("event_id").reset_index(drop=True)
        expected_cycles = int(metadata["requested_cycles"])
        embedded_config_text = json.dumps(
            metadata["config"], sort_keys=True, separators=(",", ":")
        )
        recomputed_config_hash = hashlib.sha256(
            embedded_config_text.encode("utf-8")
        ).hexdigest()
        if recomputed_config_hash != metadata.get("configuration_sha256"):
            raise RuntimeError(f"Run {run_id} has an invalid configuration hash.")
        final_cycle = int(metrics["cycle"].max())
        all_converged = bool(diagnostics["converged"].astype(bool).all())
        inverted = int(metrics["inverted_cell_count"].max())
        minimum_ratio = float(metrics["minimum_signed_jacobian_ratio"].min())
        configured_limit = float(
            metadata["config"]["solver"]["minimum_jacobian_ratio"]
        )
        expected_cycle_values = list(range(expected_cycles + 1))
        observed_cycle_values = sorted(metrics["cycle"].astype(int).tolist())
        if observed_cycle_values != expected_cycle_values:
            raise RuntimeError(
                f"Run {run_id} has cycle rows {observed_cycle_values}, "
                f"expected {expected_cycle_values}."
            )
        schedule = np.asarray(metadata["config"]["study"]["load_schedule"], dtype=float)
        if len(diagnostics) != expected_cycles * len(schedule):
            raise RuntimeError(
                f"Run {run_id} has {len(diagnostics)} solver-step rows, expected "
                f"{expected_cycles * len(schedule)}."
            )
        for cycle, group in diagnostics.sort_values("step_index").groupby("cycle"):
            if not np.allclose(group["target_fraction"].to_numpy(float), schedule):
                raise RuntimeError(
                    f"Run {run_id} cycle {cycle} does not follow the locked load schedule."
                )
        pool_columns = [
            "event_id",
            "direction_x",
            "direction_y",
            "direction_z",
            "peak_force",
        ]
        pool_array = pool.loc[:, pool_columns].to_numpy(dtype=np.float64)
        pool_signature = hashlib.sha256(pool_array.tobytes()).hexdigest()
        vector_sum = pool[["direction_x", "direction_y", "direction_z"]].to_numpy(float).sum(axis=0)
        if np.linalg.norm(vector_sum) > 1.0e-12:
            raise RuntimeError(f"Run {run_id} load pool is not antipodally balanced.")
        event_lookup = pool.set_index("event_id")
        for selected in metrics.loc[metrics["cycle"] > 0].itertuples(index=False):
            expected = event_lookup.loc[int(selected.event_id)]
            observed_values = np.asarray(
                [selected.direction_x, selected.direction_y, selected.direction_z, selected.peak_force],
                dtype=float,
            )
            expected_values = expected[
                ["direction_x", "direction_y", "direction_z", "peak_force"]
            ].to_numpy(float)
            if not np.allclose(observed_values, expected_values, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"Run {run_id} cycle {selected.cycle} does not match its load-pool event."
                )
        quality_rows.append(
            {
                "run_id": run_id,
                "pool_seed": int(metadata["pool_seed"]),
                "policy": metadata["policy"],
                "status": "complete",
                "expected_cycles": expected_cycles,
                "final_cycle": final_cycle,
                "all_load_steps_converged": all_converged,
                "inverted_cell_count": inverted,
                "minimum_signed_jacobian_ratio": minimum_ratio,
                "configured_jacobian_limit": configured_limit,
                "elapsed_seconds": float(metadata["elapsed_seconds"]),
                "mesh_cells": int(metadata["mesh"]["cells"]),
                "mesh_vertices": int(metadata["mesh"]["vertices"]),
                "configuration_sha256": metadata["configuration_sha256"],
                "pool_signature_sha256": pool_signature,
            }
        )
        if final_cycle != expected_cycles:
            raise RuntimeError(f"Run {run_id} ended at {final_cycle}, expected {expected_cycles}.")
        if not all_converged:
            raise RuntimeError(f"Run {run_id} contains a non-converged load step.")
        if inverted != 0 or minimum_ratio < configured_limit:
            raise RuntimeError(f"Run {run_id} violates mesh-validity criteria.")
        metric_frames.append(metrics)
    if not metric_frames:
        raise RuntimeError("No complete runs were found.")
    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(quality_rows)


def validate_pairing(
    metrics: pd.DataFrame,
    quality: pd.DataFrame,
    expected_pool_seeds: Sequence[int] | None = None,
    expected_config_hash: Optional[str] = None,
) -> None:
    if quality["run_id"].duplicated().any():
        raise RuntimeError("Duplicate run IDs were found in the input directory.")
    if quality["configuration_sha256"].nunique() != 1:
        raise RuntimeError("Runs do not share one locked configuration hash.")
    observed_config_hash = str(quality["configuration_sha256"].iloc[0])
    if expected_config_hash and observed_config_hash != expected_config_hash:
        raise RuntimeError(
            "Run configuration hash does not match the expected publication config."
        )
    if quality["mesh_cells"].nunique() != 1 or quality["mesh_vertices"].nunique() != 1:
        raise RuntimeError("Runs do not share one mesh topology.")
    actual_seeds = set(metrics["pool_seed"].astype(int).unique())
    if expected_pool_seeds:
        expected_seeds = set(int(seed) for seed in expected_pool_seeds)
        if actual_seeds != expected_seeds:
            missing = sorted(expected_seeds - actual_seeds)
            unexpected = sorted(actual_seeds - expected_seeds)
            raise RuntimeError(
                f"Pool-seed set mismatch; missing={missing}, unexpected={unexpected}."
            )
    final_policies = (
        metrics.groupby("pool_seed")["policy"].apply(lambda values: set(values)).to_dict()
    )
    required = set(POLICY_ORDER)
    for pool_seed, policies in final_policies.items():
        if policies != required:
            raise RuntimeError(
                f"Pool {pool_seed} has policies {sorted(policies)}, expected {POLICY_ORDER}."
            )
        signatures = quality.loc[
            quality["pool_seed"] == pool_seed, "pool_signature_sha256"
        ]
        if signatures.nunique() != 1:
            raise RuntimeError(
                f"Pool {pool_seed} policies do not contain the exact same direction/force pool."
            )
        event_sets = []
        for policy in POLICY_ORDER:
            selected = metrics.loc[
                (metrics["pool_seed"] == pool_seed)
                & (metrics["policy"] == policy)
                & (metrics["cycle"] > 0),
                "event_id",
            ]
            if selected.duplicated().any():
                raise RuntimeError(f"Pool {pool_seed}/{policy} repeats a load event.")
            event_sets.append(set(selected.astype(int)))
        if not all(event_set == event_sets[0] for event_set in event_sets[1:]):
            raise RuntimeError(f"Pool {pool_seed} policies do not share the same event set.")


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.sort_values("cycle")
        .groupby(["pool_seed", "policy"], as_index=False)
        .tail(1)
        .sort_values(["pool_seed", "policy"])
        .reset_index(drop=True)
    )


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n == 0:
        return float("nan")
    if n <= 20:
        count = 0
        total = 2**n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float(np.mean(differences * np.asarray(signs))))
            if statistic >= observed - 1.0e-15:
                count += 1
        return count / total
    rng = np.random.default_rng(20260701)
    samples = 200000
    signs = rng.choice((-1.0, 1.0), size=(samples, n))
    statistics = np.abs((signs * differences[None, :]).mean(axis=1))
    return float((np.count_nonzero(statistics >= observed) + 1) / (samples + 1))


def bootstrap_interval(differences: np.ndarray) -> Tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(20260701)
    indices = rng.integers(0, len(differences), size=(20000, len(differences)))
    means = differences[indices].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def compare_policies(final: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    difference_rows = []
    comparisons = [
        ("feedback", "random"),
        ("random", "fixed"),
        ("feedback", "fixed"),
    ]
    for metric in METRICS:
        pivot = final.pivot(index="pool_seed", columns="policy", values=metric)
        for policy_a, policy_b in comparisons:
            paired = pivot[[policy_a, policy_b]].dropna()
            differences = (paired[policy_a] - paired[policy_b]).to_numpy(float)
            if len(differences) == 0:
                continue
            ci_low, ci_high = bootstrap_interval(differences)
            standard_deviation = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
            effect_size = (
                float(differences.mean() / standard_deviation)
                if standard_deviation > 0.0
                else float("nan")
            )
            comparison = f"{policy_a}_minus_{policy_b}"
            comparison_rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "n_pairs": len(differences),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "paired_effect_dz": effect_size,
                    "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
                }
            )
            for pool_seed, difference in zip(paired.index, differences):
                difference_rows.append(
                    {
                        "pool_seed": int(pool_seed),
                        "metric": metric,
                        "comparison": comparison,
                        "difference": float(difference),
                    }
                )
    return pd.DataFrame(comparison_rows), pd.DataFrame(difference_rows)


def mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    count = int(np.count_nonzero(np.isfinite(values)))
    if count <= 1:
        return mean, mean, mean
    standard_error = float(np.nanstd(values, ddof=1) / math.sqrt(count))
    return mean, mean - 1.96 * standard_error, mean + 1.96 * standard_error


def style_axis(axis: plt.Axes, panel_label: Optional[str] = None) -> None:
    """Apply one restrained publication style without changing plotted data."""
    axis.grid(color="#D5D9DE", linewidth=0.7, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=4, width=0.8)
    if panel_label:
        axis.text(
            0.0,
            1.025,
            panel_label,
            transform=axis.transAxes,
            color=PANEL_LABEL_COLOR,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def plot_trajectories(metrics: pd.DataFrame, figure_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    plot_metrics = [
        ("shape_anisotropy", "Residual shape anisotropy"),
        ("plastic_localization_cv", "Plastic-strain localization (CV)"),
        ("plastic_strain_p95", "Equivalent plastic strain, p95"),
    ]
    for panel_label, axis, (metric, label) in zip("ABC", axes, plot_metrics):
        for policy in POLICY_ORDER:
            subset = metrics[metrics["policy"] == policy]
            grouped = subset.groupby("cycle")[metric]
            cycles = []
            means = []
            lows = []
            highs = []
            for cycle, values in grouped:
                mean, low, high = mean_ci(values.to_numpy(float))
                cycles.append(cycle)
                means.append(mean)
                lows.append(low)
                highs.append(high)
            color = POLICY_COLORS[policy]
            axis.plot(cycles, means, label=policy.title(), color=color, linewidth=2.0)
            axis.fill_between(cycles, lows, highs, color=color, alpha=0.16)
        axis.set_xlabel("Completed load-unload cycles")
        axis.set_ylabel(label)
        style_axis(axis, panel_label)
    axes[0].legend(frameon=False, loc="upper left", handlelength=2.2)
    figure.suptitle(
        "Matched-pool policy trajectories (mean and normal-approximation 95% CI)",
        fontsize=13,
        fontweight="semibold",
    )
    figure.savefig(figure_dir / "policy_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_paired_primary(final: pd.DataFrame, figure_dir: Path) -> None:
    pivot = final.pivot(index="pool_seed", columns="policy", values=PRIMARY_METRIC)
    figure, axis = plt.subplots(figsize=(7.6, 5.5), constrained_layout=True)
    x_positions = np.arange(len(POLICY_ORDER))
    for pool_seed, row in pivot.iterrows():
        values = [row[policy] for policy in POLICY_ORDER]
        axis.plot(x_positions, values, color="#A0A0A0", alpha=0.55, linewidth=1.0)
        axis.scatter(x_positions, values, color=[POLICY_COLORS[p] for p in POLICY_ORDER], s=30)
    means = [float(pivot[policy].mean()) for policy in POLICY_ORDER]
    axis.scatter(x_positions, means, color="#111111", marker="D", s=70, label="Policy mean", zorder=4)
    axis.set_xticks(x_positions, [policy.title() for policy in POLICY_ORDER])
    axis.set_ylabel("Final residual shape anisotropy")
    axis.set_title(
        "Paired final outcomes: identical load pools, different ordering policies",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    style_axis(axis)
    axis.xaxis.grid(False)
    axis.legend(frameon=False, loc="upper right", fontsize=9, handletextpad=0.6)
    figure.savefig(figure_dir / "paired_primary_outcome.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_feedback_entropy(metrics: pd.DataFrame, figure_dir: Path) -> None:
    feedback = metrics[(metrics["policy"] == "feedback") & (metrics["cycle"] > 0)]
    final_cycle = int(feedback["cycle"].max())
    feedback = feedback[feedback["cycle"] < final_cycle]
    figure, axis = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for pool_seed, subset in feedback.groupby("pool_seed"):
        axis.plot(subset["cycle"], subset["selection_entropy"], color="#B06A5B", alpha=0.28)
    grouped = feedback.groupby("cycle")["selection_entropy"].mean()
    axis.plot(grouped.index, grouped.values, color="#7A2F24", linewidth=2.5, label="Mean")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Completed load-unload cycles")
    axis.set_ylabel("Normalized selection entropy")
    axis.set_title(
        "Nonterminal geometry-feedback selections remained stochastic",
        fontsize=11,
        fontweight="semibold",
        pad=12,
    )
    style_axis(axis)
    axis.legend(frameon=False)
    figure.savefig(figure_dir / "feedback_selection_entropy.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def mesh_study(mesh_dir: Path, output_dir: Path, figure_dir: Path) -> Optional[pd.DataFrame]:
    rows = []
    for path in sorted(mesh_dir.glob("mesh*_force1000*_cycle_metrics.csv")):
        name = path.stem
        import re

        match = re.search(r"mesh(\d+)_", name)
        if not match:
            continue
        resolution = int(match.group(1))
        data = pd.read_csv(path)
        metadata_path = path.with_name(
            path.name.replace("_cycle_metrics.csv", "_metadata.json")
        )
        if not metadata_path.exists():
            raise RuntimeError(f"Mesh benchmark {path.name} is missing metadata.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        configured_resolution = int(metadata["config"]["geometry"]["mesh_resolution"])
        if metadata.get("status") != "complete" or configured_resolution != resolution:
            raise RuntimeError(
                f"Mesh benchmark {path.name} has inconsistent status/resolution metadata."
            )
        if int(data["cycle"].max()) != 2 or len(data) != 3:
            raise RuntimeError(f"Mesh benchmark {path.name} is not a complete two-cycle run.")
        if not data["all_load_steps_converged"].astype(bool).all():
            raise RuntimeError(f"Mesh benchmark {path.name} contains nonconverged steps.")
        if int(data["inverted_cell_count"].max()) != 0:
            raise RuntimeError(f"Mesh benchmark {path.name} contains inverted cells.")
        final = data.loc[data["cycle"] == data["cycle"].max()].iloc[-1]
        rows.append(
            {
                "mesh_resolution": resolution,
                "cells": int(metadata["mesh"]["cells"]),
                "shape_anisotropy": float(final["shape_anisotropy"]),
                "plastic_strain_p95": float(final["plastic_strain_p95"]),
                "minimum_signed_jacobian_ratio": float(final["minimum_signed_jacobian_ratio"]),
                "elapsed_seconds": float(final["elapsed_seconds"]),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame(rows).sort_values("mesh_resolution").drop_duplicates("mesh_resolution", keep="last")
    if set(frame["mesh_resolution"].astype(int)) != {4, 5, 6, 7, 8}:
        raise RuntimeError(
            "Locked mesh verification requires exactly resolutions 4, 5, 6, 7, and 8."
        )
    frame.to_csv(output_dir / "mesh_verification.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].plot(frame["cells"], frame["shape_anisotropy"], marker="o", color="#4C6A92")
    axes[0].set_ylabel("Final shape anisotropy")
    axes[1].plot(frame["cells"], frame["plastic_strain_p95"], marker="o", color="#B06A5B")
    axes[1].set_ylabel("Final plastic strain p95")
    for panel_label, axis in zip("AB", axes):
        axis.set_xlabel("Tetrahedral cells")
        style_axis(axis, panel_label)
    figure.suptitle("Two-cycle mesh-verification trend", fontsize=13, fontweight="semibold")
    figure.savefig(figure_dir / "mesh_verification.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return frame


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame.loc[:, columns].copy()
    def render(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value)

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    final: pd.DataFrame,
    comparisons: pd.DataFrame,
    quality: pd.DataFrame,
    mesh: Optional[pd.DataFrame],
) -> None:
    primary = comparisons[
        (comparisons["metric"] == PRIMARY_METRIC)
        & (comparisons["comparison"] == "feedback_minus_random")
    ].iloc[0]
    policy_summary = (
        final.groupby("policy")[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    lines = [
        "# Locked Ensemble Analysis",
        "",
        f"Complete paired pools: {final['pool_seed'].nunique()}",
        f"Complete runs: {len(final)}",
        "",
        "## Primary comparison",
        "",
        "Feedback minus geometry-blind random ordering for final residual shape anisotropy:",
        "",
        f"- Mean paired difference: {primary['mean_difference']:.8g}",
        f"- Bootstrap 95% CI: [{primary['ci95_low']:.8g}, {primary['ci95_high']:.8g}]",
        f"- Paired effect size dz: {primary['paired_effect_dz']:.6g}",
        f"- Two-sided exact sign-flip p: {primary['exact_sign_flip_p']:.6g}",
        "",
        "This is exploratory computational evidence, not physical validation.",
        "",
        "## All paired comparisons",
        "",
        markdown_table(
            comparisons,
            [
                "metric",
                "comparison",
                "n_pairs",
                "mean_difference",
                "ci95_low",
                "ci95_high",
                "paired_effect_dz",
                "exact_sign_flip_p",
            ],
        ),
        "",
        "## Quality gate",
        "",
        f"All load steps converged: {bool(quality['all_load_steps_converged'].fillna(False).all())}",
        f"Maximum inverted-cell count: {int(quality['inverted_cell_count'].fillna(0).max())}",
        f"Minimum signed Jacobian ratio: {quality['minimum_signed_jacobian_ratio'].min():.8g}",
        "",
    ]
    if mesh is not None:
        lines.extend(
            [
                "## Mesh-verification trend",
                "",
                markdown_table(
                    mesh,
                    [
                        "mesh_resolution",
                        "cells",
                        "shape_anisotropy",
                        "plastic_strain_p95",
                        "minimum_signed_jacobian_ratio",
                        "elapsed_seconds",
                    ],
                ),
                "",
            ]
        )
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics, quality = load_complete_runs(input_dir)
    expected_config_hash = None
    if args.expected_config:
        expected_config = json.loads(
            Path(args.expected_config).resolve().read_text(encoding="utf-8")
        )
        expected_config_text = json.dumps(
            expected_config, sort_keys=True, separators=(",", ":")
        )
        expected_config_hash = hashlib.sha256(
            expected_config_text.encode("utf-8")
        ).hexdigest()
    validate_pairing(
        metrics,
        quality,
        args.expected_pool_seeds,
        expected_config_hash,
    )
    final = final_rows(metrics)
    comparisons, differences = compare_policies(final)
    metrics.to_csv(output_dir / "all_cycle_metrics.csv", index=False)
    final.to_csv(output_dir / "final_outcomes.csv", index=False)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)
    differences.to_csv(output_dir / "paired_differences.csv", index=False)
    quality.to_csv(output_dir / "quality_summary.csv", index=False)
    plot_trajectories(metrics, figure_dir)
    plot_paired_primary(final, figure_dir)
    plot_feedback_entropy(metrics, figure_dir)
    mesh = None
    if args.mesh_study_dir:
        mesh = mesh_study(Path(args.mesh_study_dir).resolve(), output_dir, figure_dir)
    write_report(output_dir, final, comparisons, quality, mesh)
    print(f"Analyzed {len(final)} runs across {final['pool_seed'].nunique()} paired pools.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
