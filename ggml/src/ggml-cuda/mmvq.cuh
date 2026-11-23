#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Frees the Pascal Q6_K re-laid-out shadow copies whose source tensor lives in [base, base+size).
void ggml_cuda_q6k_shadow_release(const void * base, size_t size);
void ggml_cuda_q80_shadow_release(const void * base, size_t size);
void ggml_cuda_qcache_new_epoch();
bool ggml_cuda_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul_t);
bool ggml_cuda_add_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * add_t, const ggml_tensor * rms, const ggml_tensor * mul_t);
void ggml_cuda_fa_q8_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_fa_q8_finish(ggml_backend_cuda_context & ctx);
bool ggml_cuda_glu_q8_1(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
bool ggml_cuda_mul_mat_vec_q8_0_qkv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
bool ggml_cuda_node_skipped(const ggml_tensor * node);
const ggml_tensor * ggml_cuda_find_v_set_rows(const ggml_cgraph * cgraph, int from, const ggml_tensor * mm, int n_tokens);
void ggml_cuda_mark_node_skipped(const ggml_tensor * node);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
