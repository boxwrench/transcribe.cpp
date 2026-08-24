# Validation

Both stock attempts and the four-wavefront candidate completed 500 measured
iterations after 20 warmups for every frozen shape on gfx1100 and gfx1201. All
three checksum manifests pass. The stock kernel was rebuilt from restored
source before the second stock attempt, so that replication also verifies the
candidate was removed from the binaries.

GPU results agree with the CPU Q8_0 graph. Across all candidate shapes,
maximum absolute error is 0.000001 and maximum relative error is 0.000183,
inside the frozen 1e-3 gates.

The candidate failed the performance screening gate: frequency-weighted p50
improved only 1.92% on gfx1100 and regressed 20.83% on gfx1201. Consequently,
the full served benchmark and quality panel were not run. They are promotion
gates, not a way to rescue a candidate that has already hit a declared kill
condition.
