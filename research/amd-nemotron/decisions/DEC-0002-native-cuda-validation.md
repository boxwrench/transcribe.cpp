# DEC-0002: native CUDA validation gates upstream submission

EXP-0003 changes ggml's shared CUDA/HIP backend, so its current evidence proves
the promoted HIP path on gfx1100 and gfx1201 but does not prove native CUDA.

This host has no NVIDIA PCI device, `nvidia-smi`, CUDA toolkit (`nvcc`), or
native CUDA build directory. Native CUDA validation therefore cannot be run
locally. No CUDA result is inferred from HIP translation.

Before proposing the fusion upstream to ggml, run the complete two-shape
`norm-fusion-check`, repository tests, and at least one served stock/candidate
A/B on a reasonably modern NVIDIA GPU. If that resource remains unavailable,
an upstream patch must initially gate ordinary LayerNorm fusion to HIP and
leave native CUDA disabled until equivalent evidence exists. The local
promoted AMD configuration remains enabled.
