# LEAD-0005 measurement attempt 1: profiler observer effect

Decision: `HOLD_PROFILER_OBSERVER_EFFECT`.

This was diagnostic measurement, not an optimization experiment. All twelve
full-trace request timelines reconcile exactly. That validates the
interval analyzer but not the representativeness of the observation. The
graph-enabled captures contradict established unprofiled behavior:

| Target / condition | Request median | GPU-busy union | GPU non-busy | Busy/request |
| --- | ---: | ---: | ---: | ---: |
| gfx1100, graphs disabled | 1358.85 ms | 355.06 ms | 986.87 ms | 26.62% |
| gfx1201, graphs disabled | 1460.76 ms | 395.89 ms | 1064.87 ms | 27.10% |
| gfx1100, graphs enabled | 3375.63 ms | 526.21 ms | 2828.95 ms | 16.08% |
| gfx1201, graphs enabled | 2987.51 ms | 410.10 ms | 2577.41 ms | 13.73% |

The revised analyzer also recovers the following diagnostic medians. Enabled
rows remain inadmissible for production causality because of the observer
effect.

| Target / condition | Kernels | HIP calls | Graph launch / instantiate / update | Sync | Mean / p95 kernel gap | CPU blocked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gfx1100, disabled | 115,180 | 333,178 | 0 / 0 / 0 | 13,826 | 8.56 / 22.73 us | 25.01% |
| gfx1201, disabled | 114,970 | 334,372 | 0 / 0 / 0 | 13,826 | 9.26 / 26.44 us | 25.37% |
| gfx1100, enabled | 115,180 | 47,664 | 1,700 / 26 / 50 | 13,826 | 24.55 / 39.81 us | 28.00% |
| gfx1201, enabled | 114,970 | 47,959 | 1,677 / 25 / 25 | 13,826 | 22.41 / 36.44 us | 22.67% |

EXP-0001 proved that graphs improve unprofiled served latency on both targets.
Under rocprofiler, enabled graphs instead become 148% slower than disabled on
gfx1100 and 105% slower on gfx1201, and the cross-target ordering reverses.
This is not production behavior.

The observer effect is caused by kernel-dispatch tracing, not merely HIP API
tracing. A gfx1100 device-only repetition removed the HIP runtime domain but
still measured 3371.06 ms, essentially identical to the 3375.63 ms full trace.
The graph-enabled API and utilization buckets are therefore inadmissible for
causal attribution.

The graph-disabled captures remain qualified diagnostic evidence because they
preserve the production direction. Their separate-capture medians put gfx1201
101.90 ms behind, with 40.83 ms more GPU-busy union and 77.99 ms more GPU
non-busy time. Those median differences are not additive: every individual
timeline reconciles, but medians from independent runs need not.

No optimization is selected. Attempt 2 must use low-overhead in-process host
timers for graph management, graph launch, and synchronization, then validate
a GPU-activity mechanism against uninstrumented request latency before a formal
busy/request ratio is admitted.

Attempt 1 also does not yet satisfy the complete LEAD-0005 metric contract.
Rocprofiler's memory-copy trace reports direction but not byte count, and the
benchmark's frontend/decoder durations lack request-scoped start/end spans.
Therefore H2D/D2H bytes, graph instantiate/update frequency under an admissible
observer, and mutually exclusive CPU frontend/RNNT attribution remain missing.
The revised analyzer reports available kernel/API/graph/sync/event counts and
inter-kernel gap statistics and marks unavailable fields explicitly.

`EXP-0004` is unassigned. A future EXP number requires a measured mechanism and
a frozen, falsifiable intervention.
