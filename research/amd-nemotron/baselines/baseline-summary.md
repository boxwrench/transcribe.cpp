# WAVE-0001 Q8_0 baseline

These are the trained low-lookahead profiles on `jfk.wav`: R=1 for the
English model and R=3 for multilingual. Each row contains 30 measured served
streaming requests after two warmups. Raw JSON and checksum manifests live in
the corresponding timestamped local baseline directories.

| Target | Model | Mean request | p95 request | p99 feed | p95 first result |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU + BLAS | English | 2307 ms | 2417 ms | 64.2 ms | 234 ms |
| CPU + BLAS | Multilingual | 1305 ms | 1362 ms | 47.6 ms | 180 ms |
| Vulkan gfx1100 | English | 939 ms | 981 ms | 16.4 ms | 87 ms |
| Vulkan gfx1100 | Multilingual | **527 ms** | **554 ms** | **17.9 ms** | **63 ms** |
| Vulkan gfx1201 | English | 1230 ms | 1366 ms | 22.9 ms | 116 ms |
| Vulkan gfx1201 | Multilingual | 688 ms | 714 ms | 21.3 ms | 83 ms |
| HIP gfx1100 | English | **782 ms** | **840 ms** | **14.3 ms** | **80 ms** |
| HIP gfx1100 | Multilingual | 581 ms | 751 ms | 32.7 ms | 118 ms |
| HIP gfx1201 | English | 930 ms | 953 ms | 17.6 ms | 96 ms |
| HIP gfx1201 | Multilingual | 542 ms | 588 ms | 21.7 ms | 95 ms |

All feed p99 values are below the 80 ms input-feed interval, though occasional
maxima can exceed it. HIP wins English on both GPUs. Vulkan wins multilingual
on both GPUs and avoids the large HIP gfx1100 tail. The RX 7900 XT is faster
than the R9700 in every matched row on this software stack. Only Q8_0 is in
WAVE-0001, so no cross-quant winner is claimed.

The correctness panel produced byte-identical output within each model across
CPU, both Vulkan devices, and both isolated HIP targets for `jfk`, `dots`, and
`noise`. All ten baseline checksum manifests validate.
