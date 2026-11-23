# Pascal (GTX 1080 Ti) — experimental work-in-progress branch

Unvalidated snapshot of ongoing work (AI-assisted, Claude Code). Not a PR, not stable.
The validated subset is on branch `pascal-dp4a`.

On top of the Pascal Q4_K/Q6_K MMVQ kernels, this adds (all Pascal-oriented, see the diff):
Q8_0 kernels/shadow copy, fused rms_norm/add+norm -> Q8_1 quantization, GLU->Q8_1, QKV fusion,
Q8 flash-attention path, Pascal MMQ config tuning, rope and speculative-decoding tweaks.

Each step was validated individually during development (backups kept locally), but the combined
snapshot has not been re-run through the full test suite / KLD. Expect rough edges and extra VRAM use.
Build with `-DCMAKE_CUDA_ARCHITECTURES=61`. Use at your own risk.
