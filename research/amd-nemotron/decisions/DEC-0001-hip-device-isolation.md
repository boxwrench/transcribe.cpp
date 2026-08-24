# DEC-0001: isolate one physical GPU for each HIP target

HIP baseline and profiler runs set `HIP_VISIBLE_DEVICES=0` for the RX 7900 XT
and `HIP_VISIBLE_DEVICES=1` for the R9700, then select logical device 0. This
is a correctness qualification, not a performance optimization.

The gfx1201 binary crashed reproducibly in `ggml_cuda_op_fill` during
`ggml_cuda_graph_evaluate_and_capture` when all three AMD GPUs were visible and
the R9700 was selected as device 1. Disabling graphs did not reliably eliminate
the crash. The same offline and streaming commands succeeded with graph capture
enabled when `HIP_VISIBLE_DEVICES=1` exposed only the R9700 as logical device 0.

Both HIP targets use the same one-device-at-a-time rule so their measurements
have matched visibility semantics. Vulkan retains its resolved native indices.
LEAD-0001 tracks the unresolved mixed-architecture device-registry mechanism.
