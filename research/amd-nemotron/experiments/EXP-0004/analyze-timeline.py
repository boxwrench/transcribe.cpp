#!/usr/bin/env python3
"""Reconcile request-scoped rocprofiler timelines without double counting."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


Interval = tuple[int, int]


def trace_file(root: Path, suffix: str) -> Path:
    matches = list(root.rglob(f"*_{suffix}.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one *_{suffix}.csv under {root}, found {len(matches)}")
    return matches[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def interval(row: dict[str, str]) -> Interval:
    return int(row["Start_Timestamp"]), int(row["End_Timestamp"])


def clip(item: Interval, scope: Interval) -> Interval | None:
    start = max(item[0], scope[0])
    end = min(item[1], scope[1])
    return (start, end) if end > start else None


def clipped(items: list[Interval], scope: Interval) -> list[Interval]:
    return [value for item in items if (value := clip(item, scope)) is not None]


def merge(items: list[Interval]) -> list[Interval]:
    out: list[list[int]] = []
    for start, end in sorted(items):
        if not out or start > out[-1][1]:
            out.append([start, end])
        else:
            out[-1][1] = max(out[-1][1], end)
    return [(start, end) for start, end in out]


def duration(items: list[Interval]) -> int:
    return sum(end - start for start, end in merge(items))


def api_bucket(name: str) -> str:
    if "Synchronize" in name:
        return "hip_sync"
    if "Graph" in name and name != "hipGraphLaunch":
        return "graph_management"
    if any(token in name for token in ("GraphLaunch", "LaunchKernel", "Memcpy", "Memset")):
        return "dispatch_memcpy_api"
    return "other_hip_api"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--bench-json", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--device-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    markers = read_rows(trace_file(args.trace_root, "marker_api_trace"))
    hip_rows = [] if args.device_only else read_rows(trace_file(args.trace_root, "hip_api_trace"))
    kernel_rows = read_rows(trace_file(args.trace_root, "kernel_trace"))
    copy_rows = read_rows(trace_file(args.trace_root, "memory_copy_trace"))
    bench = json.loads(args.bench_json.read_text(encoding="utf-8"))

    requests: list[tuple[int, Interval]] = []
    for row in markers:
        match = re.search(r"transcribe\|request\|iter=(\d+)", row.get("Function", ""))
        if match:
            requests.append((int(match.group(1)), interval(row)))
    requests.sort()
    if len(requests) != len(bench["per_iter"]):
        raise RuntimeError(f"request marker count {len(requests)} != benchmark iterations {len(bench['per_iter'])}")

    kernels = [interval(row) for row in kernel_rows]
    copies = [interval(row) for row in copy_rows]
    api_by_bucket: dict[str, list[Interval]] = defaultdict(list)
    api_all: list[Interval] = []
    for row in hip_rows:
        value = interval(row)
        api_all.append(value)
        api_by_bucket[api_bucket(row["Function"])].append(value)

    rows: list[dict[str, object]] = []
    priority = ("gpu_busy", "hip_sync", "graph_management", "dispatch_memcpy_api", "other_hip_api")
    for (iteration, scope), timing in zip(requests, bench["per_iter"], strict=True):
        request_ns = scope[1] - scope[0]
        kernel_scope = clipped(kernels, scope)
        copy_scope = clipped(copies, scope)
        gpu_scope = merge(kernel_scope + copy_scope)
        api_scope = {name: merge(clipped(values, scope)) for name, values in api_by_bucket.items()}
        api_union = merge(clipped(api_all, scope))

        critical_inputs = {"gpu_busy": gpu_scope, **api_scope}
        critical_ns: dict[str, int] = defaultdict(int)
        claimed: list[Interval] = []
        claimed_ns = 0
        for name in priority:
            claimed = merge(claimed + critical_inputs.get(name, []))
            next_claimed_ns = duration(claimed)
            critical_ns[name] = next_claimed_ns - claimed_ns
            claimed_ns = next_claimed_ns
        critical_ns["non_hip_host"] = request_ns - claimed_ns

        gpu_busy_ns = duration(gpu_scope)
        if gpu_scope:
            submission_span_ns = gpu_scope[-1][1] - gpu_scope[0][0]
            gpu_idle_inside_ns = submission_span_ns - gpu_busy_ns
        else:
            submission_span_ns = 0
            gpu_idle_inside_ns = 0
        reconciliation_ns = request_ns - sum(critical_ns.values())
        rows.append({
            "iteration": iteration,
            "request_ms": request_ns / 1e6,
            "gpu": {
                "kernel_union_ms": duration(kernel_scope) / 1e6,
                "copy_union_ms": duration(copy_scope) / 1e6,
                "busy_union_ms": gpu_busy_ns / 1e6,
                "nonbusy_ms": (request_ns - gpu_busy_ns) / 1e6,
                "submission_span_ms": submission_span_ns / 1e6,
                "idle_inside_submission_ms": gpu_idle_inside_ns / 1e6,
                "busy_over_request": gpu_busy_ns / request_ns,
            },
            "host": {
                "hip_api_union_ms": duration(api_union) / 1e6,
                "hip_api_inclusive_ms": sum(end - start for start, end in clipped(api_all, scope)) / 1e6,
                "non_hip_host_ms": (request_ns - duration(api_union)) / 1e6,
                "api_union_by_bucket_ms": {
                    name: duration(items) / 1e6 for name, items in sorted(api_scope.items())
                },
            },
            "critical_path_ms": {name: value / 1e6 for name, value in sorted(critical_ns.items())},
            "critical_path_reconciliation_ms": reconciliation_ns / 1e6,
            "reported_stage_ms": {
                "frontend_mel": timing["mel_ms"],
                "encoder_inclusive": timing["encode_ms"],
                "decoder": timing["decode_ms"],
            },
        })

    def med(path: tuple[str, ...]) -> float:
        values: list[float] = []
        for row in rows:
            value: object = row
            for key in path:
                value = value[key]  # type: ignore[index]
            values.append(float(value))
        return statistics.median(values)

    result = {
        "schema": "amd-nemotron-exp0004-timeline-v1",
        "target": args.target,
        "condition": args.condition,
        "requests": rows,
        "median": {
            "request_ms": med(("request_ms",)),
            "gpu_busy_ms": med(("gpu", "busy_union_ms")),
            "gpu_nonbusy_ms": med(("gpu", "nonbusy_ms")),
            "gpu_idle_inside_submission_ms": med(("gpu", "idle_inside_submission_ms")),
            "gpu_busy_over_request": med(("gpu", "busy_over_request")),
            "hip_api_union_ms": med(("host", "hip_api_union_ms")),
            "hip_api_inclusive_ms": med(("host", "hip_api_inclusive_ms")),
            "frontend_mel_ms": med(("reported_stage_ms", "frontend_mel")),
            "decoder_ms": med(("reported_stage_ms", "decoder")),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
