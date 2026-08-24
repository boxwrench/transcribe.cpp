# Decision: HOLD

Do not promote a graph setting from this run. Directionally, disabling graphs
was 9.7% slower on gfx1100 and 10.9% slower on gfx1201, but the external GPU and
CPU-offload workload invalidates the causal comparison and badly distorts
tails. Correctness passed.

LEAD-0002 remains the first optimization target because the original clean
baseline/profile contradiction is independent of EXP-0001. Repeat this exact
protocol in an idle window before modifying graph or kernel code.
