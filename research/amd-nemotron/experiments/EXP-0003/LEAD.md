# LEAD-0004

The graph-disabled node map finds 10,080 ordinary affine layer-normalization
chains per request. Their `NORM`, broadcast `MUL`, and broadcast `ADD` kernels
consume 71.456 ms on gfx1100 and 56.779 ms on gfx1201. ggml-cuda already fuses
the equivalent RMS normalization chain, but not ordinary layer normalization.
