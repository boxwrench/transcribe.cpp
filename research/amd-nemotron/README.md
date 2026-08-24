# AMD Nemotron streaming optimization on Linux

This campaign establishes a reproducible AMD/Linux baseline for NVIDIA's two
0.6B cache-aware Nemotron ASR models in `transcribe.cpp`:

- `nemotron-speech-streaming-en-0.6b`;
- `nemotron-3.5-asr-streaming-0.6b`.

The measured targets are a Radeon RX 7900 XT (`gfx1100`) and Radeon AI PRO
R9700 (`gfx1201`) using CPU/OpenBLAS, Vulkan/RADV, and ROCm 7.2.1 HIP builds.
Milestone 1 established correctness, controlled Q8_0 baselines, route and shape
proof, ROCm profiling, and ranked leads. The follow-on wrenchwork has now
promoted one cross-architecture HIP kernel optimization through complete
served and quality validation.

## Results

Every accelerated route produced byte-identical transcripts to its CPU
reference for `jfk.wav`, `dots.wav`, and `noise.wav`. Each build also passes
all 37 project tests.

The trained low-lookahead `jfk.wav` results use 30 measured requests after two
warmups:

| Route | English mean / p95 | Multilingual mean / p95 |
| --- | ---: | ---: |
| CPU + BLAS | 2307 / 2417 ms | 1305 / 1362 ms |
| Vulkan gfx1100 | 939 / 981 ms | **527 / 554 ms** |
| Vulkan gfx1201 | 1230 / 1366 ms | 688 / 714 ms |
| HIP gfx1100 | **782 / 840 ms** | 581 / 751 ms |
| HIP gfx1201 | 930 / 953 ms | 542 / 588 ms |

HIP is the best initial English route on both GPUs. Vulkan is the best initial
multilingual route and avoids the HIP gfx1100 tail. All measured feed p99
latencies remain below the 80 ms input cadence. See
[`baselines/baseline-summary.md`](baselines/baseline-summary.md) and the
machine-readable [`baseline-summary.csv`](baselines/baseline-summary.csv).

ROCm profiling attributes more than 70% of GPU time to named kernel families.
Q8_0 matrix-vector work is largest at 20.95% on gfx1100 and 34.39% on gfx1201.
The main contradiction is that gfx1201 has less aggregate traced kernel time
but higher served request time. Graph launch and synchronization behavior is
therefore the first cheap falsification target; skinny Q8 matvec remains the
second-ranked engineering target. Details are in
[`profiles/profile-summary.md`](profiles/profile-summary.md).

## Important HIP device rule

Use one physical HIP GPU at a time and select logical device 0:

```bash
HIP_VISIBLE_DEVICES=0 build/hip-gfx1100/bin/transcribe-cli --backend rocm --device 0 ...
HIP_VISIBLE_DEVICES=1 build/hip-gfx1201/bin/transcribe-cli --backend rocm --device 0 ...
```

The gfx1201 binary crashes in the HIP graph path when it selects physical
ordinal 1 while the mixed `gfx1100`/`gfx1201`/integrated registry is visible.
Graph-enabled offline and streaming runs are stable when the R9700 is isolated.
Disabling graphs did not reliably fix the mixed-device case. The evidence and
qualification are recorded in [`DEC-0001`](decisions/DEC-0001-hip-device-isolation.md).

## Reproduce the campaign

Install a C++ toolchain, CMake/Ninja, Vulkan development tools, ROCm 7.2.1,
OpenBLAS, `jq`, `uv`, and `hf`. ROCm profiling additionally needs
`rocprofiler-sdk` and `hsa-amd-aqlprofile`.

From the repository root:

```bash
research/amd-nemotron/scripts/capture-system.sh
research/amd-nemotron/scripts/build-all.sh
research/amd-nemotron/scripts/download-models.sh
research/amd-nemotron/scripts/list-devices.sh
research/amd-nemotron/scripts/run-correctness-matrix.sh
research/amd-nemotron/scripts/run-baseline-matrix.sh
```

Collect representative HIP traces with:

```bash
HIP_VISIBLE_DEVICES=0 research/amd-nemotron/scripts/run-profile.sh \
  hip-gfx1100 nemotron-speech-streaming-en-0.6b en 0
HIP_VISIBLE_DEVICES=1 research/amd-nemotron/scripts/run-profile.sh \
  hip-gfx1201 nemotron-speech-streaming-en-0.6b en 0
```

GPU commands require access to `/dev/kfd` and `/dev/dri`. The model manifest
pins exact filenames, sizes, and SHA-256 hashes. The device map pins PCI and
logical-device identities. Raw campaign runs are intentionally ignored because
they are large; tracked summaries, protocols, decisions, and hashes are the
reviewable evidence layer.

## Benchmark tool

`transcribe-stream-bench` is a backend-neutral served-stream benchmark added by
this campaign. It reports raw timing for every feed and per-request summaries
for:

- feed mean, p50, p95, p99, and maximum;
- first-result compute latency;
- complete request and finalization latency;
- internal mel, encoder, and decoder time;
- final transcript and token IDs.

Example:

```bash
build/hip-gfx1100/bin/transcribe-stream-bench \
  --model models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --sample samples/jfk.wav --backend rocm --device 0 --language en \
  --feed-ms 80 --att-right 1 --warmup 2 --iters 30 --json-out result.json
```

`q8-matvec-bench` is the EXP-0002 screening tool. It builds one Q8_0
`ggml_mul_mat` graph for an explicit K/M/N shape, compares GPU output with the
CPU backend, and reports per-dispatch mean/p50/p95/p99. Run it through the
experiment script so GPU isolation, graph disabling, repetitions, immutable
paths, and checksums stay consistent.

## Evidence layout

- `OBJECTIVES.md` freezes priorities, metrics, and correctness gates.
- `manifests/` records model hashes, workload, devices, waves, and shapes.
- `baselines/` and `profiles/` contain curated summaries.
- `leads/` ranks observed opportunities without assuming their mechanism.
- `experiments/EXP-0001/` contains the frozen graph A/B protocol, both attempts,
  and the authoritative clean result.
- `experiments/EXP-0002/` contains the Q8_0 geometry protocol, replicated stock
  evidence, rejected candidate evidence, and the exact killed patch.
- `phase-map/` contains the graph-disabled node-to-kernel attribution and the
  diagnostic ROCTx patch used to produce it.
- `experiments/EXP-0003/` contains the promoted affine layer-normalization
  fusion, shape-complete H1 evidence, 120-request ABBA H2 evidence, and quality
  validation.
- `experiments/EXP-0004/` freezes the measurement-only gfx1201 execution-system
  timeline experiment and its observer-effect-qualified first attempt.
- `decisions/` records qualifications that affect valid measurements.
- `ledger.jsonl` is the append-only campaign decision log.

EXP-0001 is closed as `QUALIFY`. Its first attempt remains preserved as
contaminated evidence. In the clean rerun, disabling graphs was 11.65% slower
on gfx1100 and 3.19% slower on gfx1201, with matching p95 direction and exact
transcript/token output. Graphs remain enabled. Their architecture-sensitive
benefit explains 62.198 ms of the clean cross-target gap but does not explain
the entire contradiction.

## What happens next

EXP-0002 tested four versus eight wavefronts for the dominant single-column
Q8_0 projections. The candidate was numerically correct and improved the
`1024x4096` shape by 6.92% on gfx1100 and 10.00% on gfx1201, but it regressed
the other gfx1201 production shapes by 35–38%. Its frequency-weighted gfx1201
result was 20.83% slower, confirmed by a rebuilt-stock replication, so the
decision is `KILL` and the stock eight-wavefront tables are restored.

The graph-disabled phase map found 10,080 repeated `NORM -> MUL -> ADD`
affine layer-normalization chains. EXP-0003 now fuses each eligible chain into
one kernel using the existing fused RMSNorm design as precedent. The complete
production-shape benchmark improved 25.33% on gfx1100 and 25.42% on gfx1201.
In 120-request ABBA served validation, mean latency improved 5.39% and 7.04%,
while p95 improved 3.02% and 8.96%. Numerical limits, transcript checks, the
expanded quality panel, and 37/37 repository tests pass. Decision: `PROMOTE`.
See [`experiments/EXP-0003/DECISION.md`](experiments/EXP-0003/DECISION.md).

Do not add a shape-specific four-wavefront branch. Q8_0 remains open only for
a materially different intervention; the four-wavefront geometry must not be
retried.

LEAD-0005 separately preserves the unresolved gfx1201 execution-overhead gap.
The measured graph-benefit differential explains about 61 ms of the 148 ms
production gap, leaving roughly 87 ms that aggregate kernel duration does not
explain. Phase behavior is architecture-specific: gfx1201 is slower in FFN and
convolution but faster in attention, cache/state, and elementwise/layout. This
lead is now the next diagnostic target. It must begin with a frozen experiment
that separates graph construction/replay, dispatch, synchronization, CPU, and
other non-kernel time before proposing another intervention.

The promoted state is permanently tagged `amd-nemotron-m1` at commit
`22dbe1c`. Native CUDA validation is still required before upstream submission
because this machine contains no NVIDIA device or CUDA toolkit; see
[`DEC-0002`](decisions/DEC-0002-native-cuda-validation.md).

LEAD-0006 separately preserves the deterministic long-stream RNNT numerical
divergence exposed by EXP-0003. It is important correctness evidence, but stays
deferred until the primary LEAD-0005 execution-system investigation closes.

EXP-0004 attempt 1 reconciled every captured request but found that rocprofiler
kernel tracing reverses known graph-enabled production behavior. The result is
`HOLD_PROFILER_OBSERVER_EFFECT`; graph-enabled trace ratios are not admitted.
The graph-disabled trace still places gfx1201 about 102 ms behind and shows
both a larger GPU-busy union and larger non-busy interval, so LEAD-0005 remains
open for a lower-overhead in-process measurement.

No optimization is promoted from a microbenchmark alone. Complete served
streaming requests and the frozen quality panel remain the final gates.
