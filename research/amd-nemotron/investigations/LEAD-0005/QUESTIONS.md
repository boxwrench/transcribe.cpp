# LEAD-0005 questions

Primary question:

> If graph-disabled summed GPU compute is approximately equal on gfx1100 and
> gfx1201, where does the additional gfx1201 graph-enabled served-request time
> actually go?

Candidate explanations remain questions, not interventions:

- graph creation, update, or replay is slower;
- accumulated host-to-device launch gaps leave the GPU idle;
- synchronization or event handling blocks the CPU longer;
- frontend or RNNT host decoding takes longer;
- transfer count or volume differs;
- architecture-specific GPU work invalidates the equal-compute premise under
  the graph-enabled production request.

LEAD-0005 performs measurement only. It will not receive an EXP number or make
an optimization change. The next experiment is frozen only after a reconciled,
production-admissible measurement identifies a mechanism and a falsifiable
intervention.
