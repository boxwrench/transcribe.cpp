# EXP-0004: gfx1201 execution-system timeline

Status: attempt 1 `HOLD_PROFILER_OBSERVER_EFFECT`; LEAD-0005 remains open.

This experiment measures LEAD-0005 without changing model computation. Each
post-warmup served request is delimited with ROCTx while rocprofiler records HIP
runtime, kernel, and memory-copy timestamps. Graph-enabled and graph-disabled
conditions are captured on both isolated GPUs.

Two complete decompositions prevent double counting:

1. Device view: request wall is GPU-busy union plus all non-busy request time.
2. Host view: request wall is HIP-API union plus non-HIP application time.

A third, hierarchical critical-path view assigns every request nanosecond to
exactly one bucket, preferring GPU activity before synchronization, graph
management, dispatch/memcpy, other HIP API, and finally non-HIP host work.
Inclusive HIP API durations remain available diagnostically but are never
added to overlapping GPU time.

The principal output is `GPU_busy / T_request` from matched profiled request
intervals. The experiment may nominate an intervention only after both timeline
views reconcile and a stable bucket explains at least half the cross-target
non-GPU gap.

Attempt 1 validated the reconciliation machinery but falsified rocprofiler
kernel tracing as a production-representative graph-enabled observer. See
[`ATTEMPT-1.md`](ATTEMPT-1.md). No optimization has been selected.
