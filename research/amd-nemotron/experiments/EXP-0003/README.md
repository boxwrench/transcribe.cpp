# EXP-0003: affine layer-normalization fusion

Status: `PROMOTE`.

This experiment tests a single localized change: extend ggml-cuda's existing
fused RMS-normalization structure to ordinary `NORM -> MUL -> ADD`. The frozen
shape manifest covers all 10,080 production occurrences. H1 measures
graph-disabled computation/memory behavior; H2 separately measures
graph-enabled execution structure. The protocol defines numerical,
cross-target regression, ABBA timing, quality, and decision gates. No
microbenchmark result alone can promote it.

The shape-weighted graph-disabled benchmark improved 25.33% on gfx1100 and
25.42% on gfx1201, supporting H1 with no material-shape regression. In the
120-request ABBA promotion run, complete graph-enabled served mean improved
5.39% on gfx1100 and 7.04% on gfx1201; p95 improved 3.02% and 8.96%. This
supports H2 and clears the frozen promotion rule on both targets.

Both production shapes passed the `1e-4` numerical limit. The short, clean,
and multilingual quality cases were exact. The long continuous case was exact
on gfx1100; on gfx1201 the candidate deterministically matched gfx1100 and
reduced normalized word error against the repository golden transcript from
42.07% to 37.39%. The candidate therefore passes the quality gate and becomes
the default path. See [`RESULT.json`](RESULT.json),
[`VALIDATION.md`](VALIDATION.md), and [`DECISION.md`](DECISION.md).
