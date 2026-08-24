#!/usr/bin/env python3
"""Rank repeated post-fusion ggml node chains by approximate kernel budget."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--kernel-trace", required=True, type=Path)
    parser.add_argument("--marker-trace", required=True, type=Path)
    parser.add_argument("--served-baseline-ms", required=True, type=float)
    parser.add_argument("--min-occurrences", type=int, default=1000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    kernel_ns: Counter[str] = Counter()
    with args.kernel_trace.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            name = row["Kernel_Name"]
            if name.startswith("ggml|"):
                kernel_ns[name] += int(row["End_Timestamp"]) - int(row["Start_Timestamp"])

    graphs: list[list[tuple[str, str]]] = []
    graph: list[tuple[str, str]] = []
    label_occurrences: Counter[str] = Counter()
    last_node = -1
    pattern = re.compile(r"node=(\d+)\|op=([^|]+).*?\|ne=([^|]+)")
    with args.marker_trace.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            label = row["Function"]
            match = pattern.search(label)
            if not match:
                continue
            node = int(match.group(1))
            if graph and node <= last_node:
                graphs.append(graph)
                graph = []
            graph.append((f"{match.group(2)}[{match.group(3)}]", label))
            label_occurrences[label] += 1
            last_node = node
    if graph:
        graphs.append(graph)

    average_ns = {label: elapsed / label_occurrences[label] for label, elapsed in kernel_ns.items()}
    rows: list[dict[str, object]] = []
    for length in (2, 3):
        counts: Counter[tuple[str, ...]] = Counter()
        budgets: defaultdict[tuple[str, ...], float] = defaultdict(float)
        for nodes in graphs:
            for index in range(len(nodes) - length + 1):
                window = nodes[index:index + length]
                key = tuple(item[0] for item in window)
                counts[key] += 1
                budgets[key] += sum(average_ns.get(item[1], 0.0) for item in window)
        for key, count in counts.items():
            if count < args.min_occurrences:
                continue
            kernel_ms = budgets[key] / 1e6
            rows.append({
                "length": length,
                "pattern": " -> ".join(key),
                "occurrences": count,
                "approx_kernel_ms": kernel_ms,
                "impossible_elimination_ceiling_pct": 100.0 * kernel_ms / args.served_baseline_ms,
            })

    rows.sort(key=lambda row: (-float(row["approx_kernel_ms"]), -int(row["occurrences"])))
    result = {
        "schema": "amd-nemotron-repeated-patterns-v1",
        "target": args.target,
        "served_baseline_ms": args.served_baseline_ms,
        "method": "Kernel duration is apportioned across repeated renamed node labels, then summed for each contiguous occurrence. Ceilings assume impossible full elimination and may overlap.",
        "minimum_occurrences": args.min_occurrences,
        "patterns": rows[:50],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
