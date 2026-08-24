#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {"schema": "amd-nemotron-exp0003-promotion-v1", "targets": {}}
    for arch in ("gfx1100", "gfx1201"):
        target: dict[str, object] = {}
        for condition in ("stock", "candidate"):
            files = sorted(args.attempt.glob(f"{arch}-*-{condition}.json"))
            rows = [row for path in files for row in json.loads(path.read_text())["per_iter"]]
            feeds = [feed["wall_ms"] for row in rows for feed in row["feeds"]]
            target[condition] = {
                "requests": len(rows),
                "request_wall_ms": summarize([row["request_wall_ms"] for row in rows]),
                "encode_ms": summarize([row["encode_ms"] for row in rows]),
                "feed_ms": summarize(feeds),
                "hyp_text": json.loads(files[0].read_text())["hyp_text"],
                "token_ids_csv": json.loads(files[0].read_text())["token_ids_csv"],
            }
        stock = target["stock"]["request_wall_ms"]
        candidate = target["candidate"]["request_wall_ms"]
        target["change_pct"] = {
            "mean": 100.0 * (candidate["mean"] - stock["mean"]) / stock["mean"],
            "p95": 100.0 * (candidate["p95"] - stock["p95"]) / stock["p95"],
        }
        target["exact_output_match"] = (
            target["stock"]["hyp_text"] == target["candidate"]["hyp_text"]
            and target["stock"]["token_ids_csv"] == target["candidate"]["token_ids_csv"]
        )
        result["targets"][arch] = target
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
