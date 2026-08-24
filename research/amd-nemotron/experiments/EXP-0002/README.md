# EXP-0002: narrow Q8_0 MMVQ geometry

This package is frozen before changing `ggml/src/ggml-cuda/mmvq.cu`. The
microbenchmark is a screening tool, not a promotion result. A candidate must
also pass both HIP builds, backend numerical comparison, the full project test
suite, exact served output, and the graph-enabled end-to-end latency gate.

Run the stock or candidate binaries with `run-microbench.sh <label>`. Evidence
is written to a new immutable attempt directory with a checksum manifest.
