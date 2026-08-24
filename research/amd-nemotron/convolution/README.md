# Bounded depthwise-convolution investigation

LEAD-0007 asked whether Nemotron's production depthwise convolution on HIP was
still paying an im2col/intermediate-memory cost that a direct kernel could
remove. It is not.

Both target traces contain 1,820 `CONV_2D_DW` nodes: 1,680 encoder-block and
140 pre-encode calls. Every one dispatches the existing
`conv2d_dw_kernel<float, whcn_layout>` implementation, and zero encoder
depthwise weights appear in an `IM2COL` node. This matches the current graph
builder, which selects `ggml_conv_2d_dw_direct` by default on HIP.

The fallback is retained as a useful negative control. Disabling direct
depthwise execution with `TRANSCRIBE_CONV_NO_DIRECT_DW=1` made ten-request
served mean 8.01% slower on gfx1100 and 2.64% slower on gfx1201. Transcripts
and token IDs remained exact. The desired structural win is therefore already
present in the promoted runtime.

The direct depthwise kernels themselves total only 7.07 ms on gfx1100 and
5.06 ms on gfx1201: less than a 1% impossible-elimination served ceiling. The
larger convolution-family phase also contains pointwise projections, standard
pre-encode convolution, bias/activation, and layout work; this screen found no
single replacement mechanism with a credible 3% served upside.

Decision: `PARK_ALREADY_DIRECT_NO_NEW_MECHANISM`. EXP-0004 remains unassigned.

Reproduce the fallback control by running the promoted
`transcribe-stream-bench` command once normally and once with
`TRANSCRIBE_CONV_NO_DIRECT_DW=1`, keeping model, sample, device isolation,
warmups, and iteration count identical. The exact artifact hashes and summary
statistics are frozen in `RESULT.json`.
