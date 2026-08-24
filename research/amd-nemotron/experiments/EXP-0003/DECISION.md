# EXP-0003 decision

Decision: `PROMOTE` the ordinary affine LayerNorm fusion on gfx1100 and gfx1201.

The candidate clears the frozen promotion definition on both targets:

- served mean improves 5.39% on gfx1100 and 7.04% on gfx1201;
- served p95 moves in the same direction by 3.02% and 8.96%;
- every material production shape improves in graph-disabled measurement;
- final-tensor numerical error is below `1e-4` for the complete shape set;
- transcript, token, expanded quality, and repository test gates pass.

The default build therefore enables `NORM -> MUL -> ADD` fusion. The
`GGML_CUDA_DISABLE_NORM_FUSION` compile definition remains as a narrow,
reproducible control switch; it is not enabled in production builds.

This result supports both mechanisms. H1 is demonstrated by a 24–27%
shape-weighted kernel/graph-compute reduction with graphs disabled. H2 is
demonstrated by the complete graph-enabled request improvement. It does not,
however, close the broader gfx1201 mystery: the pre-intervention graph-benefit
differential explained only about 61 ms of the 148 ms production gap. LEAD-0005
is next and must measure non-kernel, graph, dispatch, synchronization, and CPU
time before selecting another code change.
