# Hypothesis

H0: after matched request scoping and non-overlapping interval accounting, the
gfx1201 production deficit is explained by longer GPU busy time or cannot be
reconciled to a stable execution-system bucket.

H1: gfx1100 and gfx1201 perform comparable non-overlapping GPU work per served
request, but gfx1201 spends materially more request time outside GPU kernel or
copy execution.

H2: at least one mutually exclusive critical-path bucket—synchronization wait,
graph management, dispatch/memcpy API, other HIP API, or non-HIP host
execution—explains at least half of the reconciled cross-target non-GPU gap.

This is an attribution experiment, not an optimization. It must select no code
change unless the request interval, GPU interval union, and host/API interval
union independently reconcile to served wall time.
