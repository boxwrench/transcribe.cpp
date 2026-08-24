# Promoted-state quality baseline

This freezes a broader transcript-consistency baseline at runtime commit
`1a4d168`, after EXP-0003. It covers both AMD targets, both Nemotron models,
short and long English, conversational speech, noise, German, and multiple
right-lookahead settings.

All nine transcript files match byte-for-byte between gfx1100 and gfx1201:
967 words per target and zero mismatches. The noise-only sample correctly
produces an empty transcript on both. This is a deterministic regression
baseline, not a corpus-level WER claim; subsequent promoted candidates must
also pass the project's transcript and streaming/WER gates.

Reproduce each target with:

```bash
cmake -S . -B build/amd-wave1-gfx1100 -DTRANSCRIBE_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 -DTRANSCRIBE_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/amd-wave1-gfx1100 --target transcribe-cli -j
cmake -S . -B build/amd-wave1-gfx1201 -DTRANSCRIBE_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201 -DTRANSCRIBE_BUILD_TOOLS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/amd-wave1-gfx1201 --target transcribe-cli -j

research/amd-nemotron/scripts/run-promoted-quality.sh \
  amd-wave1-gfx1100 0 gfx1100 /tmp/amd-nemotron-quality-gfx1100
research/amd-nemotron/scripts/run-promoted-quality.sh \
  amd-wave1-gfx1201 1 gfx1201 /tmp/amd-nemotron-quality-gfx1201
```

The machine-readable case hashes and model identities are in `RESULT.json`.
