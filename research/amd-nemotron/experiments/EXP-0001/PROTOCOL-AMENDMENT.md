# Protocol amendment: immutable rerun evidence

Before attempt 2, `run.sh` was changed to write into a new attempt directory
and refuse to overwrite an existing directory. It also creates a SHA-256
manifest after all four arms complete.

This changes artifact handling only. The frozen workload, binaries, target
isolation, arm order, environment variables, warmups, measured iterations,
metrics, correctness gate, and decision threshold in `PROTOCOL.json` are
unchanged. The contaminated first attempt remains in `raw/`.
