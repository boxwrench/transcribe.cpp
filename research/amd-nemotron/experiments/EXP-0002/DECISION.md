# Decision: KILL

Do not change the RDNA3/RDNA4 Q8_0 single-column MMVQ geometry from eight to
four wavefronts. The candidate is numerically correct but strongly
shape-dependent. It improves the 1024x4096 projection on both GPUs, while
gfx1201 regresses 38.48% on 1024x1024 and 35.95% on 4096x1024. The weighted
gfx1201 result is 20.83% slower.

The exact rejected source change remains in `intervention.patch`; the working
tree and rebuilt binaries use the stock eight-wavefront kernel.

A shape-specific four-wavefront dispatch is also stopped here. Applying its
measured savings only to the 48 per-feed 1024x4096 calls predicts roughly
14–22 ms per complete request, below the 5% served promotion threshold before
accounting for graph overlap. A different Q8_0 mechanism may revive LEAD-0003,
but this geometry change must not.
