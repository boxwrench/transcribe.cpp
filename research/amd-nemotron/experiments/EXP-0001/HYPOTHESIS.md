# Hypothesis

H0: disabling HIP graphs changes mean served request time by less than 5% on
each target and does not explain the gfx1201 wall-time contradiction.

H1: disabling HIP graphs changes mean served request time by at least 5% on
one target, with a consistent p95/p99 direction, showing that graph/launch
scheduling is a material first optimization target.

Prediction: graphs should help this repetitive graph shape; disabling them
should increase request time. A speedup instead would identify graph replay as
avoidable work. Byte-identical transcript and token IDs are mandatory.
