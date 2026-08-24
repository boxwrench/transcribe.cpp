# Nemotron streaming ASR on AMD ROCm

This release validates Nemotron streaming speech recognition on AMD/Linux and
ships the affine LayerNorm optimization discovered during that work. The
supported evidence covers:

| GPU | Architecture | Status |
| --- | --- | --- |
| Radeon RX 7900 XT | gfx1100 (RDNA 3) | validated |
| Radeon AI PRO R9700 | gfx1201 (RDNA 4) | validated |

The validation host used Ubuntu 24.04.4 LTS, Linux 7.0.0-28, and ROCm 7.2.1.
Other ROCm-capable AMD GPUs may work, but this release does not claim they were
measured. Native CUDA is untested and outside the AMD release scope.

## Install and build

Install ROCm using AMD's current
[Radeon Linux installation guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-radeon.html),
then install the build dependencies:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build libopenblas-dev
```

Clone the AMD release and identify the architecture reported by ROCm:

```bash
git clone https://github.com/boxwrench/transcribe.cpp.git
cd transcribe.cpp
git checkout v0.2.1-amd1

/opt/rocm/bin/rocminfo | grep -o 'gfx[0-9a-f]*' | sort -u
```

Build for exactly one target. Replace `gfx1100` with `gfx1201` for an R9700:

```bash
cmake -S . -B build-amd -G Ninja \
  -DTRANSCRIBE_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DTRANSCRIBE_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-amd -j
```

Verify that the ROCm backend and intended GPU are visible:

```bash
build-amd/bin/transcribe-cli --list-devices
```

## Download Nemotron GGUF models

Install the current Hugging Face CLI if `hf` is not already available:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Download the Q8_0 English and multilingual models. These public repositories
do not require authentication:

```bash
mkdir -p models/nemotron-speech-streaming-en-0.6b
hf download handy-computer/nemotron-speech-streaming-en-0.6b-gguf \
  nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --local-dir models/nemotron-speech-streaming-en-0.6b

mkdir -p models/nemotron-3.5-asr-streaming-0.6b
hf download handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf \
  nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --local-dir models/nemotron-3.5-asr-streaming-0.6b
```

## Run streaming ASR

English, using the low-latency profile measured in Wave 1:

```bash
HIP_VISIBLE_DEVICES=0 build-amd/bin/transcribe-cli \
  --backend rocm --device 0 \
  --model models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --language en --stream-chunk-ms 80 --stream-att-right 1 \
  samples/jfk.wav
```

Multilingual German:

```bash
HIP_VISIBLE_DEVICES=0 build-amd/bin/transcribe-cli \
  --backend rocm --device 0 \
  --model models/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b-Q8_0.gguf \
  --language de-DE --stream-chunk-ms 80 --stream-att-right 3 \
  samples/german.wav
```

Input must be 16 kHz mono WAV. Convert other formats with:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Streaming settings

`--stream-att-right` selects the model's trained right-lookahead setting:

| Model | Lowest latency | Wave 1 default | Maximum-accuracy setting |
| --- | ---: | ---: | ---: |
| English Nemotron | 0 | 1 (80 ms) | 13 (1040 ms) |
| Nemotron 3.5 multilingual | 0 | 3 (240 ms) | 13 (1040 ms) |

Use the Wave 1 defaults for interactive transcription. Use 13 when accuracy
matters more than lookahead latency. The available intermediate values are
0/1/6/13 for English and 0/3/6/13 for multilingual.

### Selecting one of several AMD GPUs

When several physical HIP devices are installed, isolate the target and then
select logical device 0. For the measured machine, physical device 0 was the
gfx1100 GPU and physical device 1 was gfx1201:

```bash
HIP_VISIBLE_DEVICES=0 build-amd/bin/transcribe-cli --backend rocm --device 0 --list-devices
HIP_VISIBLE_DEVICES=1 build-amd/bin/transcribe-cli --backend rocm --device 0 --list-devices
```

Resolve the physical ordinal on your own system; do not assume this lab's
ordering. Isolation is required for the validated mixed-gfx1100/gfx1201 path.

## Benchmark

Run the same served-stream workload used for promotion checks:

```bash
HIP_VISIBLE_DEVICES=0 build-amd/bin/transcribe-stream-bench \
  --model models/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b-Q8_0.gguf \
  --sample samples/jfk.wav --backend rocm --device 0 --language en \
  --feed-ms 80 --att-right 1 --warmup 2 --iters 30 \
  --json-out amd-nemotron-benchmark.json
```

## Wave 1 results

Initial Q8_0 served baseline, measured on `jfk.wav` with two warmups and 30
requests:

| Route | English mean / p95 | Multilingual mean / p95 |
| --- | ---: | ---: |
| CPU + OpenBLAS | 2307 / 2417 ms | 1305 / 1362 ms |
| Vulkan gfx1100 | 939 / 981 ms | **527 / 554 ms** |
| Vulkan gfx1201 | 1230 / 1366 ms | 688 / 714 ms |
| HIP gfx1100 | **782 / 840 ms** | 581 / 751 ms |
| HIP gfx1201 | 930 / 953 ms | 542 / 588 ms |

HIP was the winning English route on both targets. Vulkan won the initial
multilingual comparison; this AMD release retains both backends.

The promoted `NORM -> MUL -> ADD` affine LayerNorm fusion reduced complete
graph-enabled English request latency in a 120-request-per-condition ABBA test:

| Target | Stock mean | Fused mean | Mean gain | p95 gain |
| --- | ---: | ---: | ---: | ---: |
| gfx1100 | 847.13 ms | 801.44 ms | 5.39% | 3.02% |
| gfx1201 | 983.84 ms | 914.53 ms | 7.04% | 8.96% |

The production-weighted two-shape kernel benchmark improved 25.33% on gfx1100
and 25.42% on gfx1201. Maximum absolute tensor error was `3.58e-7` against the
unfused reference.

Quality gates passed:

- 37/37 repository tests on each target build;
- exact JFK/dots/noise promotion transcripts;
- nine-case expanded panel on both GPUs and both Nemotron models;
- 967 words per GPU with zero cross-GPU transcript mismatches.

### What was investigated and rejected

| Work item | Decision | Evidence-based reason |
| --- | --- | --- |
| HIP graphs | QUALIFY | Keep enabled; disabling them slowed gfx1100 11.65% and gfx1201 3.19%. |
| Four-wave Q8_0 MMVQ | KILL | One shape improved, but production-weighted gfx1201 performance regressed 20.83%. |
| Affine LayerNorm fusion | PROMOTE | 5.39%/7.04% complete-request gain with numerical and quality gates passing. |
| gfx1201 served gap | PARK | 88.61% of the traced gap appeared as longer waits at the same 13,826 synchronization points; no safe intervention emerged. |
| FFN dispatch alternatives | PARK | Best shape-selective route projected to only 1.62% served improvement. |
| Direct depthwise convolution | ALREADY OPTIMIZED | All 1,820 depthwise nodes already use the direct kernel; remaining direct-kernel ceiling is below 1%. |

The RX 7900 XT remained faster than the R9700 on this workload even though
summed graph-disabled GPU kernel work was similar. The R9700 difference is not
described as generally slower compute: operation families trade places, and
the low-overhead investigation localized most observed wall-time loss to later
device completion at identical synchronization points.

## Research provenance

- `amd-nemotron-m1` (`22dbe1c`): first promoted optimization state.
- `amd-nemotron-wave1` (`26f5485`): completed research state with negative
  leads closed and the quality baseline frozen.
- `v0.2.1-amd1`: AMD-ready user release.

The complete protocols, raw-evidence manifests, decisions, and stopping policy
are under [`research/amd-nemotron/`](../research/amd-nemotron/README.md).

