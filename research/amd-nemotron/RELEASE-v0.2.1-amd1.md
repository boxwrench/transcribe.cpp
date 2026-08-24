# AMD Nemotron Wave 1

This fork release makes the validated Nemotron streaming ASR path available to
AMD/Linux users. It targets ROCm/HIP on:

- Radeon RX 7900 XT (`gfx1100`, RDNA 3)
- Radeon AI PRO R9700 (`gfx1201`, RDNA 4)

The promoted affine LayerNorm fusion improves complete graph-enabled English
streaming requests by 5.39% on gfx1100 and 7.04% on gfx1201. The local
production-shape benchmark improves by approximately 25.4% on both targets.

Validation includes 37/37 repository tests per target, exact promotion
transcripts, and a nine-case quality panel spanning both Nemotron models,
multiple streaming contexts, noisy and long audio, and multilingual speech.
The panel contains 967 words per GPU with zero cross-GPU transcript mismatches.

See the [AMD ROCm setup and Wave 1 results guide](https://github.com/boxwrench/transcribe.cpp/blob/v0.2.1-amd1/docs/amd-nemotron-rocm.md)
for build commands, model downloads, streaming examples, GPU selection,
benchmarking, measured results, and rejected optimization leads.

Native CUDA is untested and outside this AMD release's scope. The underlying
research checkpoint is tagged `amd-nemotron-wave1`.
