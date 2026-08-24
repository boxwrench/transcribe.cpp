# WAVE-0001 HIP profile summary

Representative graph-enabled English R=1 streaming requests were captured
with `rocprofv3 --runtime-trace --stats` on each isolated GPU. Profiling is
diagnostic only: it inflated encode wall time to 3.43 s on gfx1100 and 3.37 s
on gfx1201, versus 0.75 s and 0.89 s in the repeated unprofiled baseline.

| Kernel family | gfx1100 GPU time | gfx1201 GPU time |
| --- | ---: | ---: |
| Q8_0 matrix-vector | 20.95% | 34.39% |
| FP16 matrix-vector | 10.28% | 13.34% |
| add broadcast | 12.47% | 8.86% |
| device copy-buffer | 7.92% | 6.47% |
| F32 matrix-vector | 3.45% | 6.82% |
| normalization | 7.04% | 3.85% |
| activation quantize Q8_1 | 6.65% | 4.19% |
| multiply broadcast | 6.60% | 4.71% |

The first seven listed gfx1100 families explain 71.91% of GPU time; the first
six listed gfx1201 families explain 74.59%. This passes the named-cost gate
without pretending that profiler time is production wall time. Total traced
kernel duration was 511.260 ms on gfx1100 and 436.344 ms on gfx1201.

The HIP API trace is dominated by `hipGraphLaunch` (73.12% / 68.98%) and
`hipStreamSynchronize` (14.38% / 13.63%). These are inclusive host API
durations and overlap GPU work, so they must not be added to kernel percentages.
In unprofiled benchmark timings the encoder accounts for 95.62% of English
gfx1100 request wall time and 96.01% on gfx1201; mel is about 2.4–2.8%, and
the streaming decoder timer is below its display precision.

The main contradiction is that gfx1201 has 14.7% less total traced kernel time
but 18.8% higher unprofiled request time than gfx1100. This points first to
dispatch/graph/synchronization behavior rather than raw aggregate GPU kernel
throughput. Q8_0 matvec remains the largest individual kernel target, but the
cheaper first falsifier is the graph-enabled/disabled A/B frozen in EXP-0001.
