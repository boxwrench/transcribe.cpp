# Bounded FFN production-shape investigation

The post-EXP-0003 profile gave FFN an 11–12% impossible-elimination ceiling,
so this investigation tested whether the actual production shapes had at least
30% local headroom before assigning EXP-0004.

The material chain occurs 3,312 times:

```text
Q8_0 MUL_MAT  M=4096 N=2 K=1024
UNARY         4096x2
Q8_0 MUL_MAT  M=1024 N=2 K=4096
```

Both projections currently select Q8_0 MMVQ and dispatch one `quantize_q8_1`
kernel plus one `mul_mat_vec_q<q8_0, 2>` kernel. The stable phase map shows
that gfx1201's first projection is the anomaly: 73.05 ms versus 47.03 ms on
gfx1100. Its second projection is already faster than gfx1100.

Five interleaved, graph-disabled, exact-shape blocks compared the stock MMVQ
route with the backend's existing MMQ and hipBLAS routes. MMVQ remains best on
gfx1100. On gfx1201 MMQ improves the first projection by 22.62%, but regresses
the second by 32.23%. Selecting MMQ only for the first projection projects to
13.54% for the complete FFN chain and about 1.62% for the served request.
hipBLAS is slower on every target/shape and has larger numerical error.

Decision: `PARK_BELOW_HEADROOM_GATE`. The best observed mechanism misses both
the 30% local gate and the 3% served ceiling, so EXP-0004 remains unassigned.
The failed EXP-0002 four-wavefront geometry is not reopened. LEAD-0007 direct
depthwise convolution becomes primary.

`ffn-existing-paths.patch` is diagnostic-only: applying it adds the
`GGML_CUDA_FFN_PATH=mmq|cublas` selector used by `q8-matvec-bench`. It is not
part of the runtime and was reversed after measurement.
