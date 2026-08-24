# Graph-disabled HIP phase map

This diagnostic capture maps every dispatched kernel to its ggml node, tensor
names, operation, and shape through compile-time ROCTx ranges and
`rocprofv3 --kernel-rename`. It uses one isolated physical GPU, logical device
0, disabled HIP graphs, the English Q8_0 model, `jfk.wav`, 80 ms feeds, and
attention right context 1. Profiling time is not production latency.

The diagnostic label also records destination/source dtypes, full source
shapes, and byte strides. Those fields support exact production-shape and
backend-path investigations such as the bounded FFN screen.

| Phase | gfx1100 kernel ms / share | gfx1201 kernel ms / share |
| --- | ---: | ---: |
| FFN | 113.037 / 26.28% | 129.862 / 30.18% |
| Attention | 68.384 / 15.90% | 55.848 / 12.98% |
| Convolution | 70.952 / 16.49% | 79.128 / 18.39% |
| Cache/state | 28.750 / 6.68% | 24.107 / 5.60% |
| Elementwise/layout | 124.428 / 28.93% | 119.965 / 27.88% |
| Runtime/memory | 24.623 / 5.72% | 21.410 / 4.98% |
| Total | 430.173 | 430.319 |

Classification is exclusive and deterministic. A kernel is assigned by its
renamed tensor path in this order: cache/state, FFN, attention, convolution;
other ggml nodes are elementwise/layout and non-ggml kernels are
runtime/memory. The raw CSV traces are intentionally ignored because each pair
is about 90 MiB. [`RESULT.json`](RESULT.json) is the curated comparison.

Internal reconciliation, rather than the similar totals, validates the mapper:
all 135,710/135,501 kernel rows are assigned exactly once, category time sums
to total kernel time, no unclassified bucket remains, and both targets expose
the same 10,080 affine-normalization sequence occurrences and two production
shapes. The approximately equal 430 ms totals establish comparable aggregate
GPU work only; they are not themselves proof of correct attribution.

The traces contain 10,080 `NORM -> MUL -> ADD` affine layer-normalization
sequences. Their three kernels consume 71.456 ms on gfx1100 and 56.779 ms on
gfx1201. Relative to the unprofiled 782/930 ms served baselines, impossible
elimination ceilings are 9.14% and 6.11%. This passes the campaign's 5% gate
and freezes EXP-0003.

The complete candidate histogram is frozen in
[`EXP-0003/SHAPES.json`](../experiments/EXP-0003/SHAPES.json). The dominant
shape is contiguous `1024x2` with 9,936 occurrences; contiguous `1024x1`
accounts for the remaining 144. Both use contiguous `1024x1` gamma/beta
broadcast operands.

## Reproduce

Apply the diagnostic-only patch and build native binaries:

```bash
git apply research/amd-nemotron/phase-map/roctx-node-ranges.patch
cmake -S . -B build/profile-gfx1100 -G Ninja -DTRANSCRIBE_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 -DTRANSCRIBE_BUILD_TOOLS=ON \
  -DCMAKE_HIP_FLAGS=-DGGML_CUDA_ROCTX_PROFILE \
  '-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed /opt/rocm/lib/libroctx64.so.4'
cmake --build build/profile-gfx1100 --target transcribe-cli -j
cmake -S . -B build/profile-gfx1201 -G Ninja -DTRANSCRIBE_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 -DTRANSCRIBE_BUILD_TOOLS=ON \
  -DCMAKE_HIP_FLAGS=-DGGML_CUDA_ROCTX_PROFILE \
  '-DCMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed /opt/rocm/lib/libroctx64.so.4'
cmake --build build/profile-gfx1201 --target transcribe-cli -j
```

Capture and analyze:

```bash
research/amd-nemotron/scripts/run-phase-map.sh profile-gfx1100 0 gfx1100 782
research/amd-nemotron/scripts/run-phase-map.sh profile-gfx1201 1 gfx1201 930
git apply -R research/amd-nemotron/phase-map/roctx-node-ranges.patch
```
