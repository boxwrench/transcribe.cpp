# Hypothesis

H0: reducing the Q8_0 single-column RDNA block from eight wavefronts to four
does not improve the weighted shape microbenchmark by at least 5%, or it
regresses either target by more than 2%, or it fails the end-to-end correctness
or served-latency gate.

H1: four wavefronts reduce idle work and block reduction overhead for the
1024-wide streaming projections, improve the weighted Q8_0 microbenchmark by
at least 5% on at least one target without a greater than 2% regression on the
other, and produce at least a 5% served request improvement on one target with
matching p95 direction.

Prediction: K=1024 benefits most because the stock eight-wavefront block can
cover 64 Q8 blocks per loop trip while the row contains only 32. K=4096 is the
falsifier because it has enough work to use the wider block over multiple
trips. Exact transcript/token output and backend-op numerical agreement are
mandatory.
