# Validation

All four arms completed 30 measured requests after two warmups. `hyp_text` and
`token_ids_csv` are identical across graph settings and GPUs.

The performance evidence is not valid for promotion. During the run,
`rocm-smi --showpids` reported an unrelated `VLLM::EngineCore` on physical GPU
1 using 15–20 GB VRAM and 32–44% GPU activity. Its parent command also enabled
56 GB CPU offload, so even the gfx1100 arm cannot be treated as isolated.
Contamination is visible in gfx1100-disabled iterations 7–14 and in the first
two gfx1201-enabled iterations (8.32 s and 7.19 s).

Raw files are protected by `raw/MANIFEST.sha256`. This failed environmental
gate is preserved because deleting inconvenient evidence would make later
comparisons less trustworthy.
