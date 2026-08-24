# Decision: QUALIFY

Keep HIP graphs enabled on both targets. In the clean controlled attempt,
disabling graphs made complete request mean/p95 11.65%/11.23% slower on
gfx1100 and 3.19%/2.54% slower on gfx1201. Exact transcript and token output
passed.

The intervention itself is not an optimization. The result does qualify
LEAD-0002: graph benefit is architecture-sensitive. It saves 94.309 ms on
gfx1100 but only 32.111 ms on gfx1201, accounting for 62.198 ms of the clean
cross-target gap. That is useful diagnostic evidence, not a claim that all of
the remaining gap is graph overhead.

EXP-0001 is closed. The next optimization target is LEAD-0003, the narrow
Q8_0 matrix-vector family, with the current isolated graph-enabled HIP route as
the served end-to-end control.
