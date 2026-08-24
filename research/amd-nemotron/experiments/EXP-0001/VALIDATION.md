# Validation

Attempt 2 completed all four frozen arms with 30 measured requests after two
warmups. `hyp_text` and `token_ids_csv` are identical across graph settings and
GPUs. The per-arm JSON checksums pass against
`attempts/attempt-2-clean-hardware/MANIFEST.sha256`.

Before the run, `rocm-smi --showpids` reported no KFD processes, host load was
0.84, and the process table contained no competing workload. After the run,
ROCm again reported no KFD processes and the process table was idle. The run
therefore meets the controlled evidence goal (`C`).

The first attempt remains under `raw/` with its original checksum manifest.
It is non-authoritative unresolved (`U`) evidence because an unrelated vLLM
workload and its CPU offload contaminated the timing. It has not been deleted
or rewritten.
