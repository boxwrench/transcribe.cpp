# EXP-0004 attempt 1: profiler observer effect

Decision: `HOLD_PROFILER_OBSERVER_EFFECT`.

All twelve full-trace request timelines reconcile exactly. That validates the
interval analyzer but not the representativeness of the observation. The
graph-enabled captures contradict established unprofiled behavior:

| Target / condition | Request median | GPU-busy union | GPU non-busy | Busy/request |
| --- | ---: | ---: | ---: | ---: |
| gfx1100, graphs disabled | 1358.85 ms | 355.06 ms | 986.87 ms | 26.62% |
| gfx1201, graphs disabled | 1460.76 ms | 395.89 ms | 1064.87 ms | 27.10% |
| gfx1100, graphs enabled | 3375.63 ms | 526.21 ms | 2828.95 ms | 16.08% |
| gfx1201, graphs enabled | 2987.51 ms | 410.10 ms | 2577.41 ms | 13.73% |

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
