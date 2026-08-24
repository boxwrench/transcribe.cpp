# DEC-0002: native CUDA is outside the AMD release gate

EXP-0003 changes ggml's shared CUDA/HIP backend, so its current evidence proves
the promoted HIP path on gfx1100 and gfx1201 but does not prove native CUDA.

This host has no NVIDIA PCI device, `nvidia-smi`, CUDA toolkit (`nvcc`), or
native CUDA build directory. Native CUDA validation therefore cannot be run
locally, and no CUDA result is inferred from HIP translation.

Native CUDA is not a requirement for the AMD/Linux project or its release.
The supported evidence is explicitly ROCm/HIP on gfx1100 and gfx1201.

If EXP-0003 is proposed upstream, submit it either as a HIP-scoped change or
as shared CUDA/HIP code with an explicit statement that native CUDA is
untested. Upstream maintainers or their CI may decide whether additional CUDA
validation is needed. That decision does not block or weaken the AMD fork.
