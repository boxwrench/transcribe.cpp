# Attempt 2 run context

- Started after an explicit idle-window check.
- Pre-run host load: 0.84; no competing process in the CPU process table.
- Pre-run ROCm state: no KFD processes.
- Post-run ROCm state: no KFD processes.
- Post-run process table: idle.
- Direct GPU access was required; the preceding sandbox-only launch failed at
  model initialization and produced no result file.

This is the authoritative controlled attempt. The frozen arm order was
gfx1100 enabled, gfx1100 disabled, gfx1201 enabled, gfx1201 disabled.
