# LEAD-0005 Attempt 2: cheap host attribution

Decision: `QUALIFIED_HOLD_DEVICE_COMPLETION_WAIT`.

HIP-runtime-only tracing passed the observer gate: median overhead versus the
matched unprofiled p50 was 3.15% on gfx1100 and 2.50% on gfx1201, and the normal
cross-target ordering remained intact.

| Median metric | gfx1100 | gfx1201 | Difference |
| --- | ---: | ---: | ---: |
| Request | 856.33 ms | 1009.99 ms | +153.66 ms |
| HIP API union | 676.87 ms | 821.27 ms | +144.40 ms |
| `hipStreamSynchronize` union | 617.80 ms | 753.96 ms | +136.16 ms |
| HIP API calls | 47,664 | 47,984 | +320 |
| Stream synchronizations | 13,826 | 13,826 | 0 |
| Graph launches | 1,700 | 1,700 | 0 |
| Graph launch time | 7.28 ms | 7.26 ms | -0.03 ms |
| Graph instantiate time | 5.75 ms | 6.09 ms | +0.34 ms |
| Graph update time | 2.72 ms | 2.64 ms | -0.08 ms |
| Frontend mel | 33.24 ms | 32.30 ms | -0.94 ms |

The additional synchronization interval explains 88.61% of the traced request
gap. The count is identical, so gfx1201 is not performing more synchronization;
the host waits longer for the same device-completion points. Direct graph
management is effectively equal and CPU frontend work is slightly faster on
gfx1201. The benchmark reports RNNT decode below its current millisecond
resolution, so it cannot explain a 154 ms gap.

This answers the coarse LEAD question without pretending that synchronization
is itself a mechanism. There is no cheap, falsifiable runtime intervention to
freeze: removing waits could violate dependencies, while optimizing graph
create/update/launch has no meaningful ceiling. `EXP-0004` remains unassigned.
The lead is parked and model-level selection moves to the post-fusion profile.
