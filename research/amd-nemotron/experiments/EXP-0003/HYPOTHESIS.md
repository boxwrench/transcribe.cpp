# Hypothesis

H0: fusing ordinary `NORM -> MUL -> ADD` does not produce a repeatable
graph-disabled kernel improvement or useful graph-enabled served improvement,
regresses a material production shape or either target, changes numerical
output beyond tolerance, or changes transcripts or token IDs.

H1 — computational/memory fusion: computing mean, variance, affine scale, and
affine bias in one HIP kernel eliminates two intermediate write/read round
trips and reduces frequency-weighted graph-disabled kernel time across the
complete production shape distribution.

H2 — execution-structure fusion: replacing three graph nodes and dispatches
with one reduces launch/graph overhead and improves complete graph-enabled
served latency. This is measured separately because gfx1201's remaining
production gap is not explained by aggregate kernel duration.

Prediction: the effect is stronger on gfx1100 because the traced chain ceiling
is 9.14% there versus 6.11% on gfx1201. One dispatch per eligible chain,
shape-complete numerical agreement, and no material-shape regression are
mandatory before served timing is considered.
