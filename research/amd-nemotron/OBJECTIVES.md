# Frozen objectives

## Priority order

1. Correct streaming transcription.
2. Single-stream latency.
3. p95/p99 latency stability.
4. Realtime safety margin.
5. Resource cost.
6. Multi-stream throughput.

The primary deployment profile is one live local Linux ASR stream using Q8_0
and a low-latency trained Nemotron setting. The Radeon AI PRO R9700 (`gfx1201`)
and Radeon RX 7900 XT (`gfx1100`) are independent deployable targets and may
promote different configurations.

## Primary metrics

- per-feed p50, p95, p99, and maximum wall latency;
- first-result compute latency;
- complete-utterance wall time and realtime factor;
- transcript/token hash and WER correctness;
- GPU-active time and host/device synchronization time.

Secondary metrics are VRAM, CPU utilization, power/energy, startup/warmup cost,
and multi-stream throughput.

## Correctness gate

The CPU transcript is the canonical reference for a fixed model, quant,
language, stream profile, and audio input. An accelerated result must have the
same normalized transcript/token hash unless a pre-declared tolerance or WER
protocol explains the difference. Lower-lookahead punctuation differences
already documented by upstream are qualified, never silently accepted.

## Workload gate

The fixed initial panel is defined in
`manifests/workload/quality-panel.json`. Changing it creates a new workload
profile; it does not rewrite prior baselines.

## Non-goals for the first wave

- multi-GPU inference;
- batch-first optimization;
- a generalized lab framework, autonomous agent, MCP server, or UI;
- custom kernels before route, correctness, shape, and phase-budget gates pass.
