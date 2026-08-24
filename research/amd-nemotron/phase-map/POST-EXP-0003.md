# Post-EXP-0003 phase profile

The promoted affine LayerNorm fusion is present: the previous 10,080
`NORM -> MUL -> ADD` occurrences are now zero. This is a fresh graph-disabled
profile of the fused program, not a reuse of pre-fusion percentages.

| Phase | gfx1100 kernel ms / served ceiling | gfx1201 kernel ms / served ceiling |
| --- | ---: | ---: |
| Elementwise/layout | 148.89 / 17.32% | 122.49 / 12.37% |
| FFN | 96.89 / 11.27% | 118.71 / 11.98% |
| Convolution | 63.08 / 7.34% | 68.44 / 6.91% |
| Attention | 66.67 / 7.76% | 50.81 / 5.13% |
| Cache/state | 33.57 / 3.91% | 25.04 / 2.53% |
| Runtime/memory | 26.55 / 3.09% | 21.13 / 2.13% |

The repeated-chain screen found no pure elementwise/layout sequence with a 3%
impossible-elimination ceiling on both targets. The largest is 6,625
`ADD[1024x2] -> NORM[1024x2]` occurrences, but its ceiling is only 4.74% on
gfx1100 and 2.78% on gfx1201 before retaining either operation. It does not
justify bespoke work now.

The strongest repeated compute chain is 3,312
`MUL_MAT[4096x2] -> UNARY[4096x2] -> MUL_MAT[1024x2]` occurrences, with
impossible-elimination ceilings of 11.60% and 12.32%. Because most of that time
is matrix multiplication, it is an FFN production-shape lead, not a claim that
three kernels can be eliminated. Convolution remains the clean second family
at roughly a 7% ceiling on both targets. No intervention is frozen yet.
