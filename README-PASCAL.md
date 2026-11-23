# Pascal (GTX 1080 Ti) DP4A kernels for llama.cpp — experimental fork branch

Not a PR. AI-assisted (Claude Code) experiment, published as-is for anyone with a Pascal GPU.

Adds Pascal-tuned MMVQ kernels for Q4_K and Q6_K (single and multi-column, for speculative decoding verification) in `ggml/src/ggml-cuda/mmvq.cu`.
Q6_K uses a re-laid-out "shadow" copy of the weights (+~1.3 GB VRAM for Qwen3-8B Q4_K_M), freed with the buffer.

Measured on GTX 1080 Ti, Qwen3-8B Q4_K_M, P2 state disabled:
- tg128: 47 -> ~55-59 tok/s (+16-18 %), also at ctx 2048; CUDA graphs on/off OK
- verification n=4 tokens: 26.4 -> 19.8 ms
- test-backend-ops OK, KLD at measurement floor

Env vars (set 0 to disable): GGML_CUDA_Q4K_PASCAL, GGML_CUDA_Q6K_PASCAL, GGML_CUDA_MULTICOL.
GGML_CUDA_Q6K_NOCACHE=1 is needed to run test-backend-ops.

Build with `-DCMAKE_CUDA_ARCHITECTURES=61`. Pascal only; use at your own risk.
