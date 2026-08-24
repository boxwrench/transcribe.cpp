#!/usr/bin/env python3
"""Summarize request-scoped LEAD-0005 rocprofiler timelines."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
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
    start, end = max(item[0], scope[0]), min(item[1], scope[1])
    return (start, end) if end > start else None


def rows_in_scope(rows: list[dict[str, str]], scope: Interval) -> list[dict[str, str]]:
    return [row for row in rows if clip(interval(row), scope) is not None]


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


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def api_bucket(name: str) -> str:
    if "Synchronize" in name or "Wait" in name:
        return "sync_wait"
    if "Graph" in name and "Launch" not in name:
        return "graph_create_update"
    if "GraphLaunch" in name:
        return "graph_launch"
    if "Memcpy" in name or "Memset" in name:
        return "memcpy_memset_api"
    if "LaunchKernel" in name or "ModuleLaunch" in name:
        return "kernel_launch_api"
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
        if match := re.search(r"transcribe\|request\|iter=(\d+)", row.get("Function", "")):
            requests.append((int(match.group(1)), interval(row)))
    requests.sort()
    if len(requests) != len(bench["per_iter"]):
        raise RuntimeError(f"request markers {len(requests)} != iterations {len(bench['per_iter'])}")

    results: list[dict[str, object]] = []
    priority = ("gpu_busy", "sync_wait", "graph_create_update", "graph_launch",
                "memcpy_memset_api", "kernel_launch_api", "other_hip_api")
    for (iteration, scope), timing in zip(requests, bench["per_iter"], strict=True):
        request_ns = scope[1] - scope[0]
        kernels = rows_in_scope(kernel_rows, scope)
        copies = rows_in_scope(copy_rows, scope)
        apis = rows_in_scope(hip_rows, scope)
        kernel_spans = clipped([interval(row) for row in kernels], scope)
        copy_spans = clipped([interval(row) for row in copies], scope)
        gpu_spans = merge(kernel_spans + copy_spans)
        api_spans = clipped([interval(row) for row in apis], scope)
        by_bucket: dict[str, list[Interval]] = defaultdict(list)
        for row in apis:
            by_bucket[api_bucket(row["Function"])].append(interval(row))
        scoped_buckets = {key: merge(clipped(value, scope)) for key, value in by_bucket.items()}

        critical_inputs = {"gpu_busy": gpu_spans, **scoped_buckets}
        critical_ns: dict[str, int] = defaultdict(int)
        claimed: list[Interval] = []
        claimed_ns = 0
        for name in priority:
            claimed = merge(claimed + critical_inputs.get(name, []))
            next_ns = duration(claimed)
            critical_ns[name] = next_ns - claimed_ns
            claimed_ns = next_ns
        critical_ns["other_unattributed"] = request_ns - claimed_ns

        merged_kernels = merge(kernel_spans)
        gaps_ms = [(right[0] - left[1]) / 1e6
                   for left, right in zip(merged_kernels, merged_kernels[1:]) if right[0] > left[1]]
        gpu_busy_ns = duration(gpu_spans)
        submission_span_ns = gpu_spans[-1][1] - gpu_spans[0][0] if gpu_spans else 0
        names = Counter(row["Function"] for row in apis)
        directions = Counter(row.get("Direction", "UNKNOWN") for row in copies)
        select = lambda pattern: [row for row in apis if re.search(pattern, row["Function"])]
        graph_create = select(r"GraphCreate")
        graph_instantiate = select(r"GraphInstantiate")
        graph_update = select(r"Graph.*Update")
        graph_launch = select(r"GraphLaunch")
        sync = select(r"Synchronize|Wait")
        events = select(r"Event")
        sync_ns = duration(clipped([interval(row) for row in sync], scope))

        results.append({
            "iteration": iteration,
            "request_ms": request_ns / 1e6,
            "gpu": {
                "kernel_union_ms": duration(kernel_spans) / 1e6,
                "copy_union_ms": duration(copy_spans) / 1e6,
                "busy_union_ms": gpu_busy_ns / 1e6,
                "idle_request_ms": (request_ns - gpu_busy_ns) / 1e6,
                "submission_span_ms": submission_span_ns / 1e6,
                "idle_inside_submission_ms": (submission_span_ns - gpu_busy_ns) / 1e6,
                "busy_over_request": gpu_busy_ns / request_ns,
                "kernel_count": len(kernels),
                "memory_copy_count": len(copies),
                "memory_copy_count_by_direction": dict(sorted(directions.items())),
                "h2d_bytes": None, "d2h_bytes": None,
                "byte_count_status": "UNAVAILABLE_IN_ROCPROFV3_MEMORY_COPY_TRACE",
                "inter_kernel_gap_count": len(gaps_ms),
                "mean_inter_kernel_gap_ms": statistics.mean(gaps_ms) if gaps_ms else None,
                "p95_inter_kernel_gap_ms": percentile(gaps_ms, 0.95),
            },
            "host": {
                "hip_api_call_count": len(apis),
                "hip_api_call_count_by_function": dict(sorted(names.items())),
                "hip_api_union_ms": duration(api_spans) / 1e6,
                "hip_api_inclusive_ms": sum(end - start for start, end in api_spans) / 1e6,
                "graph_create_count": len(graph_create),
                "graph_instantiate_count": len(graph_instantiate),
                "graph_update_count": len(graph_update),
                "graph_launch_count": len(graph_launch),
                "graph_create_union_ms": duration(clipped([interval(row) for row in graph_create], scope)) / 1e6,
                "graph_instantiate_union_ms": duration(clipped([interval(row) for row in graph_instantiate], scope)) / 1e6,
                "graph_update_union_ms": duration(clipped([interval(row) for row in graph_update], scope)) / 1e6,
                "graph_launch_union_ms": duration(clipped([interval(row) for row in graph_launch], scope)) / 1e6,
                "sync_wait_count": len(sync), "event_api_count": len(events),
                "sync_wait_union_ms": sync_ns / 1e6,
                "cpu_blocked_over_request": sync_ns / request_ns,
                "api_union_by_bucket_ms": {key: duration(value) / 1e6
                                            for key, value in sorted(scoped_buckets.items())},
            },
            "critical_path_ms": {key: value / 1e6 for key, value in sorted(critical_ns.items())},
            "critical_path_reconciliation_ms": (request_ns - sum(critical_ns.values())) / 1e6,
            "reported_stage_ms": {
                "frontend_mel": timing["mel_ms"], "encoder_inclusive": timing["encode_ms"],
                "decoder": timing["decode_ms"],
                "scope_status": "DURATION_ONLY_NOT_TIMESTAMPED_FOR_CRITICAL_PATH",
            },
        })

    def med(path: tuple[str, ...]) -> float | None:
        values = []
        for row in results:
            value: object = row
            for key in path:
                value = value[key]  # type: ignore[index]
            if value is not None:
                values.append(float(value))
        return statistics.median(values) if values else None

    median_paths = {
        "request_ms": ("request_ms",), "gpu_busy_ms": ("gpu", "busy_union_ms"),
        "gpu_idle_request_ms": ("gpu", "idle_request_ms"),
        "gpu_idle_inside_submission_ms": ("gpu", "idle_inside_submission_ms"),
        "gpu_busy_over_request": ("gpu", "busy_over_request"),
        "kernel_count": ("gpu", "kernel_count"),
        "mean_inter_kernel_gap_ms": ("gpu", "mean_inter_kernel_gap_ms"),
        "p95_inter_kernel_gap_ms": ("gpu", "p95_inter_kernel_gap_ms"),
        "hip_api_call_count": ("host", "hip_api_call_count"),
        "graph_launch_count": ("host", "graph_launch_count"),
        "graph_create_count": ("host", "graph_create_count"),
        "graph_instantiate_count": ("host", "graph_instantiate_count"),
        "graph_update_count": ("host", "graph_update_count"),
        "graph_create_union_ms": ("host", "graph_create_union_ms"),
        "graph_instantiate_union_ms": ("host", "graph_instantiate_union_ms"),
        "graph_update_union_ms": ("host", "graph_update_union_ms"),
        "graph_launch_union_ms": ("host", "graph_launch_union_ms"),
        "sync_wait_count": ("host", "sync_wait_count"),
        "event_api_count": ("host", "event_api_count"),
        "hip_api_union_ms": ("host", "hip_api_union_ms"),
        "cpu_blocked_over_request": ("host", "cpu_blocked_over_request"),
        "frontend_mel_ms": ("reported_stage_ms", "frontend_mel"),
        "decoder_ms": ("reported_stage_ms", "decoder"),
    }
    output = {
        "schema": "amd-nemotron-lead0005-timeline-v2", "lead_id": "LEAD-0005",
        "classification": "DIAGNOSTIC_MEASUREMENT_NOT_EXPERIMENT",
        "target": args.target, "condition": args.condition, "requests": results,
        "median": {name: med(path) for name, path in median_paths.items()},
        "admissibility": "Requires a matched uninstrumented latency control; Attempt 1 failed this gate for graph-enabled traces.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
