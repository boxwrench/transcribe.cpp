# AMD Nemotron stopping policy

The campaign optimizes for the fat part of the curve. It does not attempt to
explain or eliminate every millisecond.

Continue when a production-observed mechanism has a believable 3–5% served
ceiling, a bounded implementation, and a complete correctness path. Prefer
repeated-work and fusion changes before architecture-specific kernels.

Stop or park a lead when its cost fragments into buckets below 20–30 ms, when
the remaining ceiling is below roughly 3%, or when the intervention cost and
maintenance burden exceed its expected served benefit. A large symptom without
a falsifiable intervention remains a LEAD; it does not automatically become an
EXP.

The first major campaign is done enough when promoted improvements survive
served and quality gates, the largest obvious repeated waste is gone, no
untested candidate has an obvious >5% ceiling, reproducible build/benchmark
instructions exist, and generally useful changes have been offered upstream.

## Current determination

The optimization-search portion of wave 1 is stopped after the bounded FFN and
depthwise-convolution screens. FFN's best existing-path candidate projects to
only 1.62% served improvement, while the desired direct depthwise path is
already enabled and its remaining kernel work has less than a 1% ceiling. No
other post-EXP-0003 lead has both an obvious >5% served opportunity and a
bounded mechanism. EXP-0004 remains unassigned.

Remaining work is release engineering rather than open-ended tuning: native
CUDA validation and upstreaming of EXP-0003, packaging/reproducibility checks,
and release documentation. Reopen optimization only when a new production
profile exposes a concrete mechanism that passes this policy.
