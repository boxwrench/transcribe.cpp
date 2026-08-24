# LEAD-0005: gfx1201 served-time reconciliation

Status: primary investigation; Attempt 1 is
`HOLD_PROFILER_OBSERVER_EFFECT`.

This is a measurement investigation, not an experiment. Its only purpose is to
locate the extra gfx1201 wall time in the same graph-enabled production request
used on gfx1100. No optimization work is allowed here, and `EXP-0004` remains
unassigned.

The required result is a request-level reconciliation of GPU work and idle
time, HIP APIs, graph create/update/launch, copies, synchronization, CPU
frontend and RNNT decode, and residual time. Counts, bytes, inter-kernel gaps,
GPU busy fraction, and CPU blocked fraction are mandatory. See
[`MEASUREMENT-PLAN.json`](MEASUREMENT-PLAN.json) for exact definitions.

Attempt 1 validated the interval arithmetic but found that rocprofiler kernel
tracing changes graph-enabled behavior drastically. Its graph-enabled buckets
are therefore not causal evidence. The graph-disabled result remains a
qualified diagnostic, and the missing byte/stage-span metrics are recorded
explicitly rather than inferred. See [`ATTEMPT-1.md`](ATTEMPT-1.md).

The investigation closes only with a production-admissible statement such as
"gfx1201 spends +X ms/request in graph replay/update" or an explicit HOLD that
states which measurement remains missing. Only then may a falsifiable
intervention be frozen as the next EXP.
