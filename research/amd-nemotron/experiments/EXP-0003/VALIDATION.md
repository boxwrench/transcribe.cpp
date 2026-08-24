# EXP-0003 validation

## Fusion and numerical gate

The debug selection run observed exactly 10,080 three-node fusions, reconciling
with the complete production manifest. `norm-fusion-check` builds the unfused
CPU and candidate GPU graphs for both production shapes and compares their
final affine-LayerNorm tensors. Both targets produced the same bounds:

| Shape | Maximum absolute error | Maximum relative error | Limit |
| --- | ---: | ---: | ---: |
| `[1024,1,1,1]` | 1.1920929e-7 | 1.94090993e-6 | 1e-4 |
| `[1024,2,1,1]` | 3.5762787e-7 | 3.70132192e-6 | 1e-4 |

The output need not be bit-identical because fusion changes rounding and may
contract the affine operation. Both shapes pass with substantial margin.

## H1: graph-disabled shape benchmark

Each stock/candidate replication used 20 warmups and 1,000 timed graph-disabled
computations per shape. The aggregate weights are the frozen 9,936/144
production occurrence counts, not an assumed representative shape.

| Target | Stock weighted mean | Candidate weighted mean | Change |
| --- | ---: | ---: | ---: |
| gfx1100 | 30.001 us | 22.403 us | -25.33% |
| gfx1201 | 28.497 us | 21.254 us | -25.42% |

Both individual shapes improved on both targets. H1 is supported.

## H2: graph-enabled served validation

The promotion run used eight balanced `A B B A A B B A` blocks, 30 requests per
block, for 120 requests per condition per target. GPUs were isolated and tested
sequentially. Negative change is faster.

| Target | Stock mean / p95 | Candidate mean / p95 | Mean change | p95 change |
| --- | ---: | ---: | ---: | ---: |
| gfx1100 | 847.128 / 948.171 ms | 801.437 / 919.542 ms | -5.39% | -3.02% |
| gfx1201 | 983.843 / 1052.224 ms | 914.534 / 957.934 ms | -7.04% | -8.96% |

The gfx1100 blocks show thermal/time drift: the first half favored the candidate
by about 8.8% and the second half by about 2.3%. The balanced schedule retains
all observations, the aggregate clears the frozen rule, and neither late half
nor p95 reverses direction. gfx1201 shows the stronger and more stable served
effect. H2 is supported, while the separate LEAD-0005 overhead question remains
open.

## Quality gate

The frozen JFK, dots, and noise checks matched exact transcript and token output
on both targets. The expanded panel mapped clean conversation to `love-loss.wav`,
continuous speech to `dots-full.wav`, and multilingual speech to `german.wav`
with `nemotron-3.5-asr-streaming-0.6b`, `de-DE`, and R=3.

- Clean conversation and multilingual output were exact stock versus candidate
  on both targets.
- Long continuous output was exact on gfx1100.
- On gfx1201, stock and candidate differed. Each result reproduced byte-for-byte
  in a second run. Against the repository golden transcript, normalized
  word-token edit rate improved from 342/813 (42.07%) to 304/813 (37.39%). The
  candidate hash matches the gfx1100 output hash.

This is a deterministic quality improvement, not a regression. The quality gate
passes without claiming internal float bit identity.

## Repository tests

Both target builds pass 37/37 tests. A first parallel invocation caused the two
`utf8_path_unit` processes to share and remove the same temporary directory;
gfx1201 passed and the isolated gfx1100 rerun passed 37/37. This is a test
isolation collision, not a candidate failure.
