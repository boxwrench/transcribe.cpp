#!/usr/bin/env python3
"""Summarize rocprofv3 node-renamed kernel traces for the Nemotron campaign."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


PHASES = (
    "FFN",
    "attention",
    "convolution",
    "cache/state",
    "elementwise/layout",
    "runtime/memory",
)


def phase_for_name(name: str) -> str:
    lowered = name.lower()
    if not lowered.startswith("ggml|"):
        return "runtime/memory"
    if "cache" in lowered or "state" in lowered:
        return "cache/state"
    if any(token in lowered for token in (".ff1.", ".ff2.", "norm_ff", "feed_forward", "ffn")):
        return "FFN"
    if any(token in lowered for token in (".attn.", "norm_attn", "attention", "self_attn", "rel_pos")):
        return "attention"
    if any(token in lowered for token in (".conv.", "norm_conv", "convolution", "pre_encode")):
        return "convolution"
    return "elementwise/layout"


def duration_ns(row: dict[str, str]) -> int:
    return int(row["End_Timestamp"]) - int(row["Start_Timestamp"])


def affine_triplet_labels(marker_trace: Path) -> tuple[set[str], int]:
    markers: list[tuple[int, int, str, str]] = []
    with marker_trace.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["Function"]
            match = re.search(r"node=(\d+).*?\|op=([^|]+)", name)
            if match:
                markers.append((int(row["Start_Timestamp"]), int(match.group(1)), match.group(2), name))
    markers.sort()

    labels: set[str] = set()
    instances = 0
    run: list[tuple[int, int, str, str]] = []
    last_node = -1

    def scan(nodes: list[tuple[int, int, str, str]]) -> int:
        found = 0
        for index in range(len(nodes) - 2):
            if [node[2] for node in nodes[index : index + 3]] == ["NORM", "MUL", "ADD"]:
                labels.update(node[3] for node in nodes[index : index + 3])
                found += 1
        return found

    for marker in markers:
        if run and marker[1] <= last_node:
            instances += scan(run)
            run = []
        run.append(marker)
        last_node = marker[1]
    instances += scan(run)
    return labels, instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--kernel-trace", required=True, type=Path)
    parser.add_argument("--marker-trace", required=True, type=Path)
    parser.add_argument("--served-baseline-ms", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    triplet_labels, triplet_instances = affine_triplet_labels(args.marker_trace)
    phase_ns: Counter[str] = Counter()
    phase_calls: Counter[str] = Counter()
    triplet_ns = 0
    triplet_calls = 0

    with args.kernel_trace.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["Kernel_Name"]
            elapsed = duration_ns(row)
            phase = phase_for_name(name)
            phase_ns[phase] += elapsed
            phase_calls[phase] += 1
            if name in triplet_labels:
                triplet_ns += elapsed
                triplet_calls += 1

    total_ns = sum(phase_ns.values())
    result = {
        "schema": "amd-nemotron-phase-map-v1",
        "target": args.target,
        "served_baseline_ms": args.served_baseline_ms,
        "total_kernel_ms": total_ns / 1e6,
        "budgets": [
            {
                "phase": phase,
                "calls": phase_calls[phase],
                "kernel_ms": phase_ns[phase] / 1e6,
                "kernel_pct": 100.0 * phase_ns[phase] / total_ns,
                "served_time_ceiling_pct": 100.0 * phase_ns[phase] / (args.served_baseline_ms * 1e6),
            }
            for phase in PHASES
        ],
        "candidate": {
            "pattern": "NORM -> MUL -> ADD affine layer normalization",
            "instances": triplet_instances,
            "kernel_calls": triplet_calls,
            "kernel_ms": triplet_ns / 1e6,
            "served_time_ceiling_pct": 100.0 * triplet_ns / (args.served_baseline_ms * 1e6),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
