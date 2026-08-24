#!/usr/bin/env python3
"""Summarize Nemotron production FFN projections from rocprof CSV traces."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


WEIGHT_RE = re.compile(r"enc\.blocks\.(\d+)\.ff([12])\.linear([12])\.weight")


def parse_marker(label):
    fields = {}
    for item in label.split("|")[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    marker_path = next(args.trace_dir.glob("*_marker_api_trace.csv"))
    kernel_path = next(args.trace_dir.glob("*_kernel_trace.csv"))

    kernels = defaultdict(list)
    with kernel_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            kernels[row["Correlation_Id"]].append(row)

    groups = defaultdict(lambda: {
        "count": 0,
        "marker_duration_ms": 0.0,
        "kernel_duration_ms": 0.0,
        "kernel_launches": 0,
        "kernels": Counter(),
        "geometries": Counter(),
        "shapes": Counter(),
        "examples": [],
    })
    with marker_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            fields = parse_marker(row["Function"])
            match = WEIGHT_RE.fullmatch(fields.get("src0", ""))
            if fields.get("op") != "MUL_MAT" or not match:
                continue
            layer, ffn_half, projection = match.groups()
            key = "first_projection" if projection == "1" else "second_projection"
            group = groups[key]
            group["count"] += 1
            m = int(fields["ne"].split("x")[0])
            n = int(fields["ne"].split("x")[1])
            k = int(fields["src0_ne"].split("x")[0])
            group["shapes"][f"M={m},N={n},K={k}"] += 1
            group["marker_duration_ms"] += (
                int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            ) / 1e6
            matched_kernels = kernels[row["Correlation_Id"]]
            group["kernel_launches"] += len(matched_kernels)
            for kernel in matched_kernels:
                duration_ms = (
                    int(kernel["End_Timestamp"]) - int(kernel["Start_Timestamp"])
                ) / 1e6
                group["kernel_duration_ms"] += duration_ms
                group["kernels"][kernel["Kernel_Name"]] += 1
                geometry = (
                    f"wg={kernel['Workgroup_Size_X']}x{kernel['Workgroup_Size_Y']}x"
                    f"{kernel['Workgroup_Size_Z']};grid={kernel['Grid_Size_X']}x"
                    f"{kernel['Grid_Size_Y']}x{kernel['Grid_Size_Z']}"
                )
                group["geometries"][geometry] += 1
            if len(group["examples"]) < 2:
                group["examples"].append({
                    "layer": int(layer),
                    "ffn_half": int(ffn_half),
                    "M_N_K": [int(fields["ne"].split("x")[0]),
                                int(fields["ne"].split("x")[1]),
                                int(fields["src0_ne"].split("x")[0])],
                    "dst_type": fields.get("dst_type"),
                    "weight_type": fields.get("src0_type"),
                    "input_type": fields.get("src1_type"),
                    "weight_shape": fields.get("src0_ne"),
                    "input_shape": fields.get("src1_ne"),
                    "weight_strides_bytes": fields.get("src0_nb"),
                    "input_strides_bytes": fields.get("src1_nb"),
                })

    result = {"gpu": args.gpu, "source": str(args.trace_dir), "projections": {}}
    for key, group in sorted(groups.items()):
        result["projections"][key] = {
            "count": group["count"],
            "marker_duration_ms": round(group["marker_duration_ms"], 6),
            "kernel_duration_ms": round(group["kernel_duration_ms"], 6),
            "kernel_launches": group["kernel_launches"],
            "kernels": dict(group["kernels"].most_common()),
            "geometries": dict(group["geometries"].most_common()),
            "shape_counts": dict(group["shapes"].most_common()),
            "examples": group["examples"],
        }

    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
