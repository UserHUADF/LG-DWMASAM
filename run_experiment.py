#!/usr/bin/env python3
"""Run the LG-DWMASAM temporal edge-importance experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys


def ensure_numpy_runtime() -> None:
    """Re-run with the bundled Codex Python when system python lacks numpy."""

    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        bundled_python = Path.home() / (
            ".cache/codex-runtimes/codex-primary-runtime/"
            "dependencies/python/bin/python3"
        )
        if bundled_python.exists() and Path(sys.executable).resolve() != bundled_python:
            os.execv(str(bundled_python), [str(bundled_python), *sys.argv])
        message = (
            "Missing dependency: numpy. Install it with `python3 -m pip install -r "
            "requirements.txt`, or run this script with the Codex bundled Python."
        )
        raise SystemExit(message)


ensure_numpy_runtime()

from lg_dwmasam import read_temporal_csv, run_lg_dwmasam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LG-DWMASAM on a temporal directed weighted edge CSV."
    )
    parser.add_argument(
        "--input",
        default="data/sample_temporal_edges.csv",
        help="CSV with columns time, source, target, weight.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for summary.json and edge_scores.csv.",
    )
    parser.add_argument(
        "--operator",
        choices=("product", "harmonic", "geometric"),
        default="product",
        help="Line-graph weight mapping operator.",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Formula (2-8) alpha.")
    parser.add_argument(
        "--kshell-q",
        type=float,
        default=1.0,
        help="Q value used by the K-shell neighborhood formula.",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.2,
        help="Top edge ratio removed in the LCC deletion test.",
    )
    parser.add_argument(
        "--seed-ratio",
        type=float,
        default=0.2,
        help="Top edge ratio used as cascade seeds.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Linear-threshold cascade activation threshold.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable max normalization before CV-weighted fusion.",
    )
    return parser.parse_args()


def write_edge_scores(path: Path, result) -> None:
    network = result.network
    rank_by_edge = {edge_index: rank + 1 for rank, edge_index in enumerate(result.ranking)}
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "rank",
            "edge_id",
            "source",
            "target",
            "aggregate_score",
            *[f"score_t{time}" for time in network.times],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge_index in result.ranking:
            edge = network.edges[edge_index]
            row = {
                "rank": rank_by_edge[edge_index],
                "edge_id": edge_index,
                "source": edge.source,
                "target": edge.target,
                "aggregate_score": f"{result.aggregate_scores[edge_index]:.12g}",
            }
            for t, time in enumerate(network.times):
                row[f"score_t{time}"] = f"{result.scores_by_time[t, edge_index]:.12g}"
            writer.writerow(row)


def write_summary(path: Path, result, args: argparse.Namespace) -> None:
    network = result.network
    top_edges = []
    for edge_index in result.ranking[: min(10, network.edge_count)]:
        edge = network.edges[edge_index]
        top_edges.append(
            {
                "edge_id": edge_index,
                "edge": edge.label,
                "score": float(result.aggregate_scores[edge_index]),
            }
        )

    summary = {
        "input": args.input,
        "operator": args.operator,
        "alpha": args.alpha,
        "kshell_q": args.kshell_q,
        "time_count": network.time_count,
        "node_count": network.node_count,
        "edge_count": network.edge_count,
        "asam_shape": list(result.asam.shape),
        "eigenvalue": result.eigenvalue,
        "monotonicity": result.monotonicity,
        "lcc_before": result.lcc_before,
        "lcc_after": result.lcc_after,
        "deletion_drop_rate": result.deletion_drop_rate,
        "cv_weights_by_time": {
            layer.time: {
                "TCM": layer.cv_weights[0],
                "SK": layer.cv_weights[1],
                "C": layer.cv_weights[2],
            }
            for layer in result.layers
        },
        "cascade_by_time": list(result.cascade_by_time),
        "top_edges": top_edges,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    network = read_temporal_csv(args.input)
    result = run_lg_dwmasam(
        network,
        operator=args.operator,
        alpha=args.alpha,
        kshell_q=args.kshell_q,
        normalize_matrices=not args.no_normalize,
        top_ratio=args.top_ratio,
        cascade_seed_ratio=args.seed_ratio,
        cascade_threshold=args.threshold,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_edge_scores(output_dir / "edge_scores.csv", result)
    write_summary(output_dir / "summary.json", result, args)

    best_edge = network.edges[result.ranking[0]]
    print("LG-DWMASAM experiment finished")
    print(f"Input: {args.input}")
    print(f"ASAM shape: {result.asam.shape[0]} x {result.asam.shape[1]}")
    print(f"Top edge: {best_edge.label}")
    print(f"Monotonicity: {result.monotonicity:.6f}")
    print(f"Deletion drop rate: {result.deletion_drop_rate:.6f}")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
