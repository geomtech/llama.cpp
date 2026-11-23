#include "mmvq.cuh"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdint>
#include <type_traits>

// only enabled on DGX Spark, where it is a gain on every type below. On the higher-bandwidth parts the kernel
// has little exposed latency left to hide and the extra requests cost more than they save.
// For perf data, see https://github.com/ggml-org/llama.cpp/pull/26705#issuecomment-5569335031
#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
// returns true only for those quants that benefit from prefetch and false otherwise
static constexpr __host__ __device__ bool mmvq_should_prefetch(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}
#endif

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA1,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10,
    MMVQ_PARAMETERS_ORIN
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA1)
    return MMVQ_PARAMETERS_RDNA1;
#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA4_M)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_GB10
    return MMVQ_PARAMETERS_GB10;
#elif __CUDA_ARCH__ == GGML_CUDA_CC_ORIN
    return MMVQ_PARAMETERS_ORIN;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA1(cc)) {
        return MMVQ_PARAMETERS_RDNA1;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_GB10) {
        return MMVQ_PARAMETERS_GB10;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_ORIN) {
        return MMVQ_PARAMETERS_ORIN;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: P100, Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        cc = ggml_cuda_highest_compiled_arch(cc);
        if (cc == GGML_CUDA_CC_PASCAL || cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4_M(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type)) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_GB10) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ORIN) {
        switch (type) { // tuned for Jetson Orin
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return ne11 <= 1;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_VOLTA) {
        switch (type) {
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
                return ne11 <= 5;
            case GGML_TYPE_Q5_K:
                return ne11 <= 6;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        switch (type) { // tuned on Radeon 8060S (gfx1151)
            case GGML_TYPE_Q8_0:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3) || defined(RDNA4_M)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif !defined(GGML_USE_MUSA) && (__CUDA_ARCH__ == GGML_CUDA_CC_PASCAL || __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
                return 2;
            case 2:
            case 3:
                return type == GGML_TYPE_Q3_K ? 1 : 2;
            case 4:
                return type == GGML_TYPE_Q6_K? 1 : 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA1) {
        switch (ncols_dst) {
            case 1:
                return 1;
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        return calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
    }
    return 1;
}

// Low bit types load few weight bytes per thread, a second row per block shares the q8_1 loads and adds memory parallelism.
static constexpr __host__ __device__ bool mmvq_two_rows_ncols1(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

static constexpr __host__ __device__ int calc_rows_per_block(ggml_type type, int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10 || table_id == MMVQ_PARAMETERS_RDNA1) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                if (table_id == MMVQ_PARAMETERS_GB10) {
                    return type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ? 2 : 1;
                }
                return (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_RDNA1) && mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_ORIN) {
        switch (ncols_dst) {
            case 1:
                if (small_k) {
                    return nwarps;
                }
                // One row per warp (more rows per block, more loads in flight) for these types, measured on Jetson Orin Nano.
                // Other types are faster in isolated mat-vec too, but gained at most 2 % end to end.
                switch (type) {
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ4_NL:
                        return nwarps;
                    default:
                        break;
                }
                return mmvq_two_rows_ncols1(type) ? 2 : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false>
__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = shared_expert ? 0 : ncols_dst == 1 && ids ? ids[channel_dst] : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const float * x_scale = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op active_glu;
    float glu_limit = 0.0f;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        glu_limit     = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
            x_scale        = (const float *) fusion.x_scale;
            gate_scale     = (const float *) fusion.gate_scale;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = x_scale[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = gate_scale[ids ? channel_x : 0];
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#if __CUDA_ARCH__ == GGML_CUDA_CC_GB10
        // start the next iterations' weight loads early
        if constexpr (mmvq_should_prefetch(type)) {
            // loop iterations, not blocks; q3_K and q6_K keep the distance in blocks they had with vdr 1
            constexpr int pf_dist = type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q6_K ? 1 : 2;
            const int kbx_pf = kbx + pf_dist*blocks_per_iter;
            if (kbx_pf < blocks_per_row_x) {
#pragma unroll
                for (int i = 0; i < rows_per_cuda_block; ++i) {
                    const size_t off = (size_t)(kbx_offset + i*stride_row_x + kbx_pf) * ggml_cuda_type_traits<type>::bs;
                    ggml_cuda_prefetch_l2((const char *) vx + off);
                    if constexpr (has_fusion) {
                        if (use_gate) {
                            ggml_cuda_prefetch_l2((const char *) vgate + off);
                        }
                    }
                }
            }
        }
#endif

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += vec_dot_q_cuda(
                            vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            case GGML_GLU_OP_SWIGLU_CLAMP:
                                result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, glu_limit, gate_bias, x_bias, x_scale, gate_scale, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion = false>
__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mm_fusion_args_device fusion,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const bool shared_expert = has_fusion && fusion.shared_up && blockIdx.y == gridDim.y - 1;
    if (shared_expert) {
        vx = fusion.shared_up;
        dst = fusion.shared_dst;
        stride_col_dst = fusion.shared_stride_col_dst;
    }

    // fuse gate, bias, scales, and glu_op into the up projection
    bool use_gate = false;
    const void  * vgate      = nullptr;
    const float * x_bias     = nullptr;
    const float * gate_bias  = nullptr;
    const float * x_scale    = nullptr;
    const float * gate_scale = nullptr;
    ggml_glu_op   active_glu = GGML_GLU_OP_SWIGLU;
    float         glu_limit  = 0.0f;

    if constexpr (has_fusion) {
        use_gate   = fusion.gate != nullptr;
        vgate      = shared_expert ? fusion.shared_gate : fusion.gate;
        x_bias     = (const float *) fusion.x_bias;
        gate_bias  = (const float *) fusion.gate_bias;
        active_glu = fusion.glu_op;
        glu_limit  = fusion.glu_limit;
        if constexpr (type == GGML_TYPE_NVFP4) {
            x_scale    = (const float *) fusion.x_scale;
            gate_scale = (const float *) fusion.gate_scale;
        }
    }

    const uint32_t token_idx   = threadIdx.y;
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    const uint32_t channel_dst = shared_expert ? 0 : blockIdx.y;

    if (token_idx >= ncols_dst) {
        return;
    }

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = shared_expert ? 0 : ids[channel_dst + token_idx * ids_stride];
    const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[i] += vec_dot_q_cuda(vgate, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            if (use_gate) {
                tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
            }
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            const uint32_t bias_idx = channel_x*stride_channel_dst + row0 + threadIdx.x;

            if constexpr (type == GGML_TYPE_NVFP4) {
                if (x_scale) {
                    result *= x_scale[channel_x];
                }
            }
            if (x_bias) {
                result += x_bias[bias_idx];
            }
            if (use_gate) {
                float gate_value = tmp_gate[threadIdx.x];
                if constexpr (type == GGML_TYPE_NVFP4) {
                    if (gate_scale) {
                        gate_value *= gate_scale[channel_x];
                    }
                }
                if (gate_bias) {
                    gate_value += gate_bias[bias_idx];
                }
                switch (active_glu) {
                    case GGML_GLU_OP_SWIGLU:
                        result *= ggml_cuda_op_silu_single(gate_value);
                        break;
                    case GGML_GLU_OP_GEGLU:
                        result *= ggml_cuda_op_gelu_single(gate_value);
                        break;
                    case GGML_GLU_OP_SWIGLU_OAI:
                        result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                        break;
                    case GGML_GLU_OP_SWIGLU_CLAMP:
                        result = ggml_cuda_op_swiglu_clamp_single(gate_value, result, glu_limit);
                        break;
                    default:
                        result = result * gate_value;
                        break;
                }
            }
        }
        dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, tmp_gate, vgate, x_bias, gate_bias, active_glu, glu_limit, x_scale, gate_scale);
    } else if constexpr (type != GGML_TYPE_NVFP4) {
        GGML_UNUSED_VARS(x_scale, gate_scale);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(type, ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;
    if constexpr (c_ncols_dst == 1) {
        if (has_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_fusion && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst + (fusion.shared_up != nullptr));
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
                            fusion.x_scale != nullptr || fusion.gate_scale != nullptr;

    if (has_fusion) {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, true>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, false>, launch_params,
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride);
    }
}

// ---------------------------------------------------------------------------------------------
// Pascal-optimised Q4_K x Q8_1 matvec (ncols_dst == 1): native ggml layout, 128-bit weight loads,
// one warp per R rows, activations staged once per block in shared memory, optional fused gate/GLU.
// ---------------------------------------------------------------------------------------------
template <int W, int R, bool GATE>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q4_K_pascal(
        const uint8_t * __restrict__ vx, const uint8_t * __restrict__ vgate, const int * __restrict__ y,
        float * __restrict__ dst, const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        const ggml_glu_op glu_op, const float glu_limit, const int ncols) {
    constexpr int NG = GATE ? 2 : 1;
    extern __shared__ int q4k_ys[];
    const int nsb  = ncols >> 8;
    const int nint = (ncols >> 5) * 9;
    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x + threadIdx.y * 32; i < nint; i += W * 32) {
        q4k_ys[i] = y[i];
    }
    __syncthreads();

    const int  lane = threadIdx.x, warp = threadIdx.y;
    const int  l8 = lane & 7, sbl = lane >> 3;
    const int  g = l8 >> 1, h = l8 & 1;
    const bool hiScale = g >= 2;
    const int  row0 = (blockIdx.x * W + warp) * R;
    const size_t rowBytes = (size_t) nsb * 144;

    float acc[NG][R];
#pragma unroll
    for (int t = 0; t < NG; ++t) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[t][r] = 0.f;
    }

    for (int sb = sbl; sb < nsb; sb += 4) {
        uint4 hd[NG][R], qq[NG][R];
#pragma unroll
        for (int t = 0; t < NG; ++t) {
            const uint8_t * base = t == 0 ? vx : vgate;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 * blk = (const uint4 *) (base + (size_t) (row0 + r) * rowBytes + (size_t) sb * 144);
                hd[t][r] = __ldg(blk);
                qq[t][r] = __ldg(blk + 1 + l8);
            }
        }
        const int * ylo = q4k_ys + (sb * 8 + 2 * g) * 9;
        const int * yhi = ylo + 9;
        const int dlo = ylo[0], dhi = yhi[0];
        const float2 flo = __half22float2(*(const half2 *) &dlo);
        const float2 fhi = __half22float2(*(const half2 *) &dhi);
        int alo[4], ahi[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { alo[i] = ylo[1 + 4 * h + i]; ahi[i] = yhi[1 + 4 * h + i]; }

#pragma unroll
        for (int t = 0; t < NG; ++t) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float2   dd  = __half22float2(*(const half2 *) &hd[t][r].x);
                const uint64_t lo8 = ((uint64_t) hd[t][r].z << 32) | hd[t][r].y;
                const uint32_t w2  = hd[t][r].w;
                uint32_t sc[2], mn[2];
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int      j  = 2 * g + k, jj = j & 3;
                    const uint32_t bj = (uint32_t) (lo8 >> (8 * j)) & 0xFF;
                    const uint32_t bm = (uint32_t) (lo8 >> (8 * (j + 4))) & 0xFF;
                    const uint32_t bl = (uint32_t) (lo8 >> (8 * jj)) & 0xFF;
                    const uint32_t b2 = (w2 >> (8 * jj)) & 0xFF;
                    const uint32_t scA = bj & 63, mA = bm & 63;
                    const uint32_t scB = (b2 & 0xF) | ((bl >> 6) << 4), mB = (b2 >> 4) | ((bj >> 6) << 4);
                    sc[k] = hiScale ? scB : scA;
                    mn[k] = hiScale ? mB : mA;
                }
                const uint32_t qv[4] = {qq[t][r].x, qq[t][r].y, qq[t][r].z, qq[t][r].w};
                int dl = 0, dh = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    dl = ggml_cuda_dp4a((int) (qv[i] & 0x0F0F0F0Fu), alo[i], dl);
                    dh = ggml_cuda_dp4a((int) ((qv[i] >> 4) & 0x0F0F0F0Fu), ahi[i], dh);
                }
                float v = dd.x * (flo.x * (float) ((int) sc[0] * dl) + fhi.x * (float) ((int) sc[1] * dh));
                if (h == 0) v -= dd.y * (flo.y * (float) mn[0] + fhi.y * (float) mn[1]);
                acc[t][r] += v;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < NG; ++t) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[t][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t][r] = v;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            float result = acc[0][r];
            if (x_bias) result += x_bias[row];
            if constexpr (GATE) {
                float gv = acc[1][r];
                if (gate_bias) gv += gate_bias[row];
                switch (glu_op) {
                    case GGML_GLU_OP_SWIGLU:       result *= ggml_cuda_op_silu_single(gv); break;
                    case GGML_GLU_OP_GEGLU:        result *= ggml_cuda_op_gelu_single(gv); break;
                    case GGML_GLU_OP_SWIGLU_OAI:   result = ggml_cuda_op_swiglu_oai_single(gv, result); break;
                    case GGML_GLU_OP_SWIGLU_CLAMP: result = ggml_cuda_op_swiglu_clamp_single(gv, result, glu_limit); break;
                    default:                       result = result * gv; break;
                }
            }
            dst[row] = result;
        }
    }
}

// Returns true when the Pascal Q4_K fast path handled the call.
static bool mul_mat_vec_q4_K_pascal_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_Q4K_PASCAL"); return !e || atoi(e) != 0; }();
    if (!enabled || ids || ncols_dst != 1 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 256 != 0 || stride_row_x != ncols_x / 256) return false;
    if (fusion.shared_up || fusion.x_scale || fusion.gate_scale) return false;
    const size_t smem = (size_t) (ncols_x / 32) * 36;
    if (smem > 48 * 1024) return false;
    if (((uintptr_t) vx & 15) || (fusion.gate && ((uintptr_t) fusion.gate & 15))) return false;

    const bool gate = fusion.gate != nullptr;
    constexpr int W = 8;
    const int R = gate ? 1 : 2;
    if (nrows_x % (W * R) != 0) return false;
    const dim3 grid(nrows_x / (W * R)), block(32, W);
    const uint8_t * x = (const uint8_t *) vx;
    const uint8_t * gt = (const uint8_t *) fusion.gate;
    const float * xb = (const float *) fusion.x_bias;
    const float * gb = gate ? (const float *) fusion.gate_bias : nullptr;
    if (gate) {
        mul_mat_vec_q4_K_pascal<W, 1, true><<<grid, block, smem, stream>>>(
            x, gt, (const int *) vy, dst, xb, gb, fusion.glu_op, fusion.glu_limit, ncols_x);
    } else {
        mul_mat_vec_q4_K_pascal<W, 2, false><<<grid, block, smem, stream>>>(
            x, nullptr, (const int *) vy, dst, xb, nullptr, fusion.glu_op, fusion.glu_limit, ncols_x);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Pascal-optimised Q6_K matvec (ncols_dst == 1). Q6_K superblocks are 210 B (not 16 B aligned), so
// each tensor is re-laid-out once into aligned SoA streams (ql / qh / scales / d) kept as a shadow copy.
// ---------------------------------------------------------------------------------------------
#include <map>
#include <mutex>

struct q6k_shadow_t { uint8_t * ql; uint8_t * qh; int8_t * sc; uint8_t * d; };

static std::map<std::pair<const void *, size_t>, q6k_shadow_t> & q6k_shadows() {
    static std::map<std::pair<const void *, size_t>, q6k_shadow_t> m;
    return m;
}
static std::mutex q6k_mtx;

// Called when a CUDA weight buffer is destroyed: drop every shadow whose source tensor lived inside it,
// so a reused address can never serve a stale copy.
void ggml_cuda_q6k_shadow_release(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(q6k_mtx);
    auto & map = q6k_shadows();
    bool synced = false;
    for (auto it = map.begin(); it != map.end();) {
        const char * p = (const char *) it->first.first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            if (!synced) { cudaDeviceSynchronize(); synced = true; }
            if (it->second.ql) { cudaFree(it->second.ql); cudaFree(it->second.qh); cudaFree(it->second.sc); cudaFree(it->second.d); }
            it = map.erase(it);
        } else {
            ++it;
        }
    }
}

static __global__ void q6k_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ ql,
        uint8_t * __restrict__ qh, int8_t * __restrict__ sc, uint8_t * __restrict__ d) {
    const size_t   sb = blockIdx.x;
    const uint8_t * s = src + sb * 210;
    for (int i = threadIdx.x; i < 210; i += blockDim.x) {
        const uint8_t v = s[i];
        if (i < 128)      ql[sb * 128 + i] = v;
        else if (i < 192) qh[sb * 64 + (i - 128)] = v;
        else if (i < 208) sc[sb * 16 + (i - 192)] = (int8_t) v;
        else              d[sb * 2 + (i - 208)] = v;
    }
}

template <int W, int R, int MINB = 1>
static __global__ void __launch_bounds__(W * 32, MINB)
mul_mat_vec_q6_K_pascal(
        const uint8_t * __restrict__ QL, const uint8_t * __restrict__ QH, const int8_t * __restrict__ SC,
        const half * __restrict__ D, const int * __restrict__ y, float * __restrict__ dst,
        const float * __restrict__ x_bias, const int ncols) {
    extern __shared__ int q6k_ys[];
    const int nsb = ncols >> 8, nblk = ncols >> 5, nint = nblk * 9;
    int * sums = q6k_ys + nint;
    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x + threadIdx.y * 32; i < nint; i += W * 32) q6k_ys[i] = y[i];
    __syncthreads();
    for (int i = threadIdx.x + threadIdx.y * 32; i < nblk * 2; i += W * 32) {
        const int * q = q6k_ys + (i >> 1) * 9 + 1 + 4 * (i & 1);
        int s = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) s = ggml_cuda_dp4a(0x01010101, q[k], s);
        sums[i] = s;
    }
    __syncthreads();

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int t = lane & 3, sbl = lane >> 2, n = t >> 1, c = t & 1;
    const int row0 = (blockIdx.x * W + warp) * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;

    for (int sb = sbl; sb < nsb; sb += 8) {
        uint4 A[R], B[R], C[R], S[R];
        half  dh[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t rs = (size_t) (row0 + r) * nsb + sb;
            A[r] = __ldg((const uint4 *) (QL + rs * 128 + 64 * n + 16 * c));
            B[r] = __ldg((const uint4 *) (QL + rs * 128 + 64 * n + 32 + 16 * c));
            C[r] = __ldg((const uint4 *) (QH + rs * 64 + 32 * n + 16 * c));
            S[r] = __ldg((const uint4 *) (SC + rs * 16));
            dh[r] = D[rs];
        }
        const int bi0 = sb * 8 + 4 * n;
        int   a[4][4];
        float d8[4];
        int   sm[4];
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const int * yb = q6k_ys + (bi0 + m) * 9;
            const int   w0 = yb[0];
            d8[m] = __low2float(*(const half2 *) &w0);
#pragma unroll
            for (int i = 0; i < 4; ++i) a[m][i] = yb[1 + 4 * c + i];
            sm[m] = sums[(bi0 + m) * 2 + c];
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint32_t wlo = n ? S[r].z : S[r].x, whi = n ? S[r].w : S[r].y;
            const uint32_t Aw[4] = {A[r].x, A[r].y, A[r].z, A[r].w};
            const uint32_t Bw[4] = {B[r].x, B[r].y, B[r].z, B[r].w};
            const uint32_t Cw[4] = {C[r].x, C[r].y, C[r].z, C[r].w};
            float tot = 0.f;
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const uint32_t w  = (m >> 1) ? whi : wlo;
                const int      sh = 8 * (c + 2 * (m & 1));
                const int      sc = ((int) (w << (24 - sh))) >> 24;
                int dot = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t lo = (m & 1) ? Bw[i] : Aw[i];
                    lo = (m >> 1) ? (lo >> 4) : lo;
                    const uint32_t hi = (Cw[i] >> (2 * m)) & 0x03030303u;
                    dot = ggml_cuda_dp4a((int) ((lo & 0x0F0F0F0Fu) | (hi << 4)), a[m][i], dot);
                }
                tot += d8[m] * (float) (sc * (dot - 32 * sm[m]));
            }
            acc[r] += __half2float(dh[r]) * tot;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) dst[row0 + r] = x_bias ? v + x_bias[row0 + r] : v;
    }
}

// Returns the re-laid-out shadow copy of a Q6_K tensor, creating it on first use (never during graph capture,
// and only when enough VRAM is free). false = caller must fall back to the stock kernel.
static bool q6k_get_shadow(const void * vx, const size_t nblocks, cudaStream_t stream, q6k_shadow_t & sh) {
    std::lock_guard<std::mutex> lock(q6k_mtx);
    auto & map = q6k_shadows();
    auto it = map.find({vx, nblocks});
    static const bool nocache = [] { const char * e = getenv("GGML_CUDA_Q6K_NOCACHE"); return e && atoi(e) != 0; }();
    if (nocache && it != map.end()) {  // debug/test mode: tensor memory may be rewritten, rebuild every call
        cudaDeviceSynchronize();
        if (it->second.ql) { cudaFree(it->second.ql); cudaFree(it->second.qh); cudaFree(it->second.sc); cudaFree(it->second.d); }
        map.erase(it);
        it = map.end();
    }
    if (it == map.end()) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return false;  // never allocate while a CUDA graph is being captured
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t need = nblocks * 210;
        if (free_b < need + (size_t) 768 * 1024 * 1024) {
            map[{vx, nblocks}] = {nullptr, nullptr, nullptr, nullptr};  // no room: remember and fall back
            return false;
        }
        q6k_shadow_t s{};
        if (cudaMalloc(&s.ql, nblocks * 128) != cudaSuccess || cudaMalloc(&s.qh, nblocks * 64) != cudaSuccess ||
            cudaMalloc(&s.sc, nblocks * 16) != cudaSuccess || cudaMalloc((void **) &s.d, nblocks * 2) != cudaSuccess) {
            cudaGetLastError();
            map[{vx, nblocks}] = {nullptr, nullptr, nullptr, nullptr};
            return false;
        }
        q6k_repack_kernel<<<(unsigned) nblocks, 128, 0, stream>>>((const uint8_t *) vx, s.ql, s.qh, s.sc, s.d);
        CUDA_CHECK(cudaGetLastError());
        it = map.emplace(std::make_pair(vx, nblocks), s).first;
    }
    sh = it->second;
    return sh.ql != nullptr;
}

static bool mul_mat_vec_q6_K_pascal_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_Q6K_PASCAL"); return !e || atoi(e) != 0; }();
    if (!enabled || ids || ncols_dst != 1 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 256 != 0 || stride_row_x != ncols_x / 256) return false;
    if (fusion.shared_up || fusion.x_scale || fusion.gate_scale || fusion.gate) return false;
    const size_t smem = (size_t) (ncols_x / 32) * 36 + (size_t) (ncols_x / 16) * 4;
    if (smem > 48 * 1024) return false;
    int W, R;
    if (nrows_x % 16 == 0)     { W = 4; R = 4; }
    else if (nrows_x % 8 == 0) { W = 8; R = 1; }
    else return false;

    const size_t nblocks = (size_t) nrows_x * (ncols_x / 256);
    q6k_shadow_t sh;
    if (!q6k_get_shadow(vx, nblocks, stream, sh)) return false;

    // Launch variants (GGML_CUDA_Q6K_VARIANT): 0 = 4x4, 1 = 4x4 capped at 4 blocks/SM,
    // 2 = 4x2 (default, 91 regs, measured ~2.4 % faster per call under Nsight), 3 = 8x2. Falls back to 8x1 when the row count does not divide.
    static const int variant = [] { const char * e = getenv("GGML_CUDA_Q6K_VARIANT"); return e ? atoi(e) : 2; }();
    const float * xb = (const float *) fusion.x_bias;
    const half *  dh = (const half *) sh.d;
    const int *   yq = (const int *) vy;
    if (variant == 1 && nrows_x % 16 == 0) {
        mul_mat_vec_q6_K_pascal<4, 4, 4><<<dim3(nrows_x / 16), dim3(32, 4), smem, stream>>>(sh.ql, sh.qh, sh.sc, dh, yq, dst, xb, ncols_x);
    } else if (variant == 2 && nrows_x % 8 == 0) {
        mul_mat_vec_q6_K_pascal<4, 2, 1><<<dim3(nrows_x / 8), dim3(32, 4), smem, stream>>>(sh.ql, sh.qh, sh.sc, dh, yq, dst, xb, ncols_x);
    } else if (variant == 3 && nrows_x % 16 == 0) {
        mul_mat_vec_q6_K_pascal<8, 2, 1><<<dim3(nrows_x / 16), dim3(32, 8), smem, stream>>>(sh.ql, sh.qh, sh.sc, dh, yq, dst, xb, ncols_x);
    } else if (W == 4) {
        mul_mat_vec_q6_K_pascal<4, 4, 1><<<dim3(nrows_x / 16), dim3(32, 4), smem, stream>>>(sh.ql, sh.qh, sh.sc, dh, yq, dst, xb, ncols_x);
    } else {
        mul_mat_vec_q6_K_pascal<8, 1, 1><<<dim3(nrows_x / 8), dim3(32, 8), smem, stream>>>(sh.ql, sh.qh, sh.sc, dh, yq, dst, xb, ncols_x);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Multi-column (2..8 tokens) Q4_K x Q8_1 matvec for Pascal. Weights are decoded once per 16-byte slice and
// reused for every column. The activations of all N columns are staged in shared memory in an aligned,
// bank-conflict-free layout, K-chunked so that N*chunk always fits in 46 KB.
// ---------------------------------------------------------------------------------------------
template <int W, int R, int N>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q4_K_pascal_mc(
        const uint8_t * __restrict__ vx, const int * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int csb, const int stride_col_y, const int stride_col_dst) {
    extern __shared__ __align__(16) uint8_t mc_smem[];
    const int nsb  = ncols >> 8;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = lane + warp * 32;
    const int l8 = lane & 7, sbl = lane >> 3, g = l8 >> 1, h = l8 & 1;
    const bool hiScale = g >= 2;
    const int  row0 = (blockIdx.x * W + warp) * R;
    const size_t rowBytes = (size_t) nsb * 144;
    const int colStride = csb * 288;   // bytes per column in smem: qs (csb*256) then ds (csb*32)

    float acc[N][R];
#pragma unroll
    for (int c = 0; c < N; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[c][r] = 0.f;
    }

    ggml_cuda_pdl_sync();
    for (int c0 = 0; c0 < nsb; c0 += csb) {
        const int cnt  = min(csb, nsb - c0);
        const int nblk = cnt * 8;
        if (c0 > 0) __syncthreads();
        for (int it = tid; it < N * nblk; it += W * 32) {
            const int col = it / nblk, b = it - col * nblk;
            const int * src = y + ((size_t) col * stride_col_y + (size_t) c0 * 8 + b) * 9;
            const int w0 = src[0];
            int q[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) q[i] = src[1 + i];
            int4 * yq = (int4 *) (mc_smem + (size_t) col * colStride);
            const int sw = ((b >> 2) & 1) << 1;
            yq[(b * 2 + 0) ^ sw] = make_int4(q[0], q[1], q[2], q[3]);
            yq[(b * 2 + 1) ^ sw] = make_int4(q[4], q[5], q[6], q[7]);
            ((int *) (mc_smem + (size_t) col * colStride + csb * 256))[b] = w0;
        }
        __syncthreads();

        for (int sbi = sbl; sbi < cnt; sbi += 4) {
            const int sb = c0 + sbi;
            uint4 hd[R], qq[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint4 * blk = (const uint4 *) (vx + (size_t) (row0 + r) * rowBytes + (size_t) sb * 144);
                hd[r] = __ldg(blk);
                qq[r] = __ldg(blk + 1 + l8);
            }
            // decode weights once (shared by all columns)
            int   ql[R][4], qh[R][4];
            float dx[R], dm[R];
            uint32_t sc0[R], sc1[R], m0[R], m1[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float2   dd  = __half22float2(*(const half2 *) &hd[r].x);
                dx[r] = dd.x; dm[r] = dd.y;
                const uint64_t lo8 = ((uint64_t) hd[r].z << 32) | hd[r].y;
                const uint32_t w2  = hd[r].w;
                uint32_t sc[2], mn[2];
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int      j  = 2 * g + k, jj = j & 3;
                    const uint32_t bj = (uint32_t) (lo8 >> (8 * j)) & 0xFF;
                    const uint32_t bm = (uint32_t) (lo8 >> (8 * (j + 4))) & 0xFF;
                    const uint32_t bl = (uint32_t) (lo8 >> (8 * jj)) & 0xFF;
                    const uint32_t b2 = (w2 >> (8 * jj)) & 0xFF;
                    const uint32_t scA = bj & 63, mA = bm & 63;
                    const uint32_t scB = (b2 & 0xF) | ((bl >> 6) << 4), mB = (b2 >> 4) | ((bj >> 6) << 4);
                    sc[k] = hiScale ? scB : scA;
                    mn[k] = hiScale ? mB : mA;
                }
                sc0[r] = sc[0]; sc1[r] = sc[1]; m0[r] = mn[0]; m1[r] = mn[1];
                const uint32_t qv[4] = {qq[r].x, qq[r].y, qq[r].z, qq[r].w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ql[r][i] = (int) (qv[i] & 0x0F0F0F0Fu);
                    qh[r][i] = (int) ((qv[i] >> 4) & 0x0F0F0F0Fu);
                }
            }
            const int blo = sbi * 8 + 2 * g;                 // local block index of the low sub-block
            const int sw  = ((blo >> 2) & 1) << 1;           // same swizzle for blo and blo+1
#pragma unroll
            for (int c = 0; c < N; ++c) {
                const int4 * yq = (const int4 *) (mc_smem + (size_t) c * colStride);
                const int4 alo = yq[(blo * 2 + h) ^ sw];
                const int4 ahi = yq[((blo + 1) * 2 + h) ^ sw];
                const int2 dsp = ((const int2 *) (mc_smem + (size_t) c * colStride + csb * 256))[blo >> 1];
                const float2 flo = __half22float2(*(const half2 *) &dsp.x);
                const float2 fhi = __half22float2(*(const half2 *) &dsp.y);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int dl = ggml_cuda_dp4a(ql[r][0], alo.x, 0);
                    dl = ggml_cuda_dp4a(ql[r][1], alo.y, dl);
                    dl = ggml_cuda_dp4a(ql[r][2], alo.z, dl);
                    dl = ggml_cuda_dp4a(ql[r][3], alo.w, dl);
                    int dh = ggml_cuda_dp4a(qh[r][0], ahi.x, 0);
                    dh = ggml_cuda_dp4a(qh[r][1], ahi.y, dh);
                    dh = ggml_cuda_dp4a(qh[r][2], ahi.z, dh);
                    dh = ggml_cuda_dp4a(qh[r][3], ahi.w, dh);
                    float v = dx[r] * (flo.x * (float) ((int) sc0[r] * dl) + fhi.x * (float) ((int) sc1[r] * dh));
                    if (h == 0) v -= dm[r] * (flo.y * (float) m0[r] + fhi.y * (float) m1[r]);
                    acc[c][r] += v;
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < N; ++c) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[c][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) dst[(size_t) c * stride_col_dst + row0 + r] = v;
        }
    }
}

template <int N>
static void mul_mat_vec_q4_K_pascal_mc_launch(const uint8_t * vx, const int * y, float * dst, int ncols, int nrows,
        int stride_col_y, int stride_col_dst, int W, int R, cudaStream_t stream) {
    const int nsb = ncols / 256;
    const int csb = std::max(1, std::min(nsb, 46000 / (288 * N)));
    const size_t smem = (size_t) N * csb * 288;
#define Q4MC(WW, RR) mul_mat_vec_q4_K_pascal_mc<WW, RR, N><<<dim3(nrows / (WW * RR)), dim3(32, WW), smem, stream>>>( \
        vx, y, dst, ncols, csb, stride_col_y, stride_col_dst)
    if (W == 4 && R == 2)      { Q4MC(4, 2); }
    else if (W == 8 && R == 1) { Q4MC(8, 1); }
    else if (W == 8 && R == 2) { Q4MC(8, 2); }
    else if (W == 8 && R == 4) { Q4MC(8, 4); }
    else if (W == 16 && R == 1) { Q4MC(16, 1); }
    else if (W == 16 && R == 2) { Q4MC(16, 2); }
    else                       { Q4MC(4, 1); }
#undef Q4MC
}

// Returns true when the multi-column Pascal Q4_K path handled the call.
static bool mul_mat_vec_q4_K_pascal_mc_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_MULTICOL"); return !e || atoi(e) != 0; }();
    if (!enabled || ids || ncols_dst < 2 || ncols_dst > 8 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 256 != 0 || stride_row_x != ncols_x / 256) return false;
    if (fusion.gate || fusion.x_bias || fusion.shared_up || fusion.x_scale || fusion.gate_scale) return false;
    if (((uintptr_t) vx & 15)) return false;
    static const int cfg_def = [] { const char * e = getenv("GGML_CUDA_MC_CFG"); return e ? atoi(e) : 2; }();
    static const long long small_elems = [] { const char * e = getenv("GGML_CUDA_MC_SMALL_ELEMS"); return e ? atoll(e) : 0LL; }();
    static const int cfg_small = [] { const char * e = getenv("GGML_CUDA_MC_SMALL_CFG"); return e ? atoi(e) : 2; }();
    const int cfg = (small_elems > 0 && (long long) nrows_x * ncols_x <= small_elems) ? cfg_small : cfg_def;
    int W = 8, R = 1;
    if (cfg == 1) { W = 4; R = 2; } else if (cfg == 2) { W = 8; R = 2; } else if (cfg == 3) { W = 4; R = 1; } else if (cfg == 4) { W = 8; R = 4; } else if (cfg == 5) { W = 16; R = 1; } else if (cfg == 6) { W = 16; R = 2; }
    if (nrows_x % (W * R) != 0) return false;
    const uint8_t * x = (const uint8_t *) vx;
    const int *     y = (const int *) vy;
    switch (ncols_dst) {
        case 2: mul_mat_vec_q4_K_pascal_mc_launch<2>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 3: mul_mat_vec_q4_K_pascal_mc_launch<3>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 4: mul_mat_vec_q4_K_pascal_mc_launch<4>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 5: mul_mat_vec_q4_K_pascal_mc_launch<5>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 6: mul_mat_vec_q4_K_pascal_mc_launch<6>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 7: mul_mat_vec_q4_K_pascal_mc_launch<7>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        default: mul_mat_vec_q4_K_pascal_mc_launch<8>(x, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Multi-column (2..8 tokens) Q6_K matvec on the re-laid-out shadow weights. Weight bits are unpacked once per
// slice and reused by every column; activations (qs + scale + per-16 sums) are staged K-chunked in shared memory.
// ---------------------------------------------------------------------------------------------
template <int W, int R, int N>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q6_K_pascal_mc(
        const uint8_t * __restrict__ QL, const uint8_t * __restrict__ QH, const int8_t * __restrict__ SC,
        const half * __restrict__ D, const int * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int csb, const int stride_col_y, const int stride_col_dst) {
    extern __shared__ __align__(16) uint8_t mc6_smem[];
    const int nsb  = ncols >> 8;
    const int lane = threadIdx.x, warp = threadIdx.y, tid = lane + warp * 32;
    const int t = lane & 3, sbl = lane >> 2, n = t >> 1, c = t & 1;
    const int row0 = (blockIdx.x * W + warp) * R;
    const int colStride = csb * 384;   // per column: qs (csb*256 B) then meta {d8, sum0, sum1, 0} (csb*128 B)

    float acc[N][R];
#pragma unroll
    for (int k = 0; k < N; ++k) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[k][r] = 0.f;
    }

    ggml_cuda_pdl_sync();
    for (int c0 = 0; c0 < nsb; c0 += csb) {
        const int cnt  = min(csb, nsb - c0);
        const int nblk = cnt * 8;
        if (c0 > 0) __syncthreads();
        for (int it = tid; it < N * nblk; it += W * 32) {
            const int col = it / nblk, b = it - col * nblk;
            const int * src = y + ((size_t) col * stride_col_y + (size_t) c0 * 8 + b) * 9;
            const int w0 = src[0];
            int q[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) q[i] = src[1 + i];
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s0 = ggml_cuda_dp4a(0x01010101, q[i], s0);
                s1 = ggml_cuda_dp4a(0x01010101, q[4 + i], s1);
            }
            int4 * yq = (int4 *) (mc6_smem + (size_t) col * colStride);
            const int swq = (((b >> 3) & 1) << 2) | (((b >> 2) & 1) << 1);
            yq[(b * 2 + 0) ^ swq] = make_int4(q[0], q[1], q[2], q[3]);
            yq[(b * 2 + 1) ^ swq] = make_int4(q[4], q[5], q[6], q[7]);
            int4 * meta = (int4 *) (mc6_smem + (size_t) col * colStride + csb * 256);
            meta[b ^ (((b >> 3) & 1) << 1)] = make_int4(__float_as_int(__low2float(*(const half2 *) &w0)), s0, s1, 0);
        }
        __syncthreads();

        for (int sbi = sbl; sbi < cnt; sbi += 8) {
            const int sb = c0 + sbi;
            int   qm[R][4][4];
            int   scm[R][4];
            float dw[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t rs = (size_t) (row0 + r) * nsb + sb;
                const uint4 A = __ldg((const uint4 *) (QL + rs * 128 + 64 * n + 16 * c));
                const uint4 B = __ldg((const uint4 *) (QL + rs * 128 + 64 * n + 32 + 16 * c));
                const uint4 C = __ldg((const uint4 *) (QH + rs * 64 + 32 * n + 16 * c));
                const uint4 S = __ldg((const uint4 *) (SC + rs * 16));
                dw[r] = __half2float(D[rs]);
                const uint32_t wlo = n ? S.z : S.x, whi = n ? S.w : S.y;
                const uint32_t Aw[4] = {A.x, A.y, A.z, A.w};
                const uint32_t Bw[4] = {B.x, B.y, B.z, B.w};
                const uint32_t Cw[4] = {C.x, C.y, C.z, C.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const uint32_t w  = (m >> 1) ? whi : wlo;
                    const int      sh = 8 * (c + 2 * (m & 1));
                    scm[r][m] = ((int) (w << (24 - sh))) >> 24;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        uint32_t lo = (m & 1) ? Bw[i] : Aw[i];
                        lo = (m >> 1) ? (lo >> 4) : lo;
                        const uint32_t hi = (Cw[i] >> (2 * m)) & 0x03030303u;
                        qm[r][m][i] = (int) ((lo & 0x0F0F0F0Fu) | (hi << 4));
                    }
                }
            }
            float tot[N][R];
#pragma unroll
            for (int k = 0; k < N; ++k) {
#pragma unroll
                for (int r = 0; r < R; ++r) tot[k][r] = 0.f;
            }
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int b   = sbi * 8 + 4 * n + m;
                const int pq  = (b * 2 + c) ^ ((((b >> 3) & 1) << 2) | (((b >> 2) & 1) << 1));
                const int pm  = b ^ (((b >> 3) & 1) << 1);
#pragma unroll
                for (int k = 0; k < N; ++k) {
                    const int4 a  = ((const int4 *) (mc6_smem + (size_t) k * colStride))[pq];
                    const int4 mt = ((const int4 *) (mc6_smem + (size_t) k * colStride + csb * 256))[pm];
                    const float d8 = __int_as_float(mt.x);
                    const int   sm = c ? mt.z : mt.y;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        int dot = ggml_cuda_dp4a(qm[r][m][0], a.x, 0);
                        dot = ggml_cuda_dp4a(qm[r][m][1], a.y, dot);
                        dot = ggml_cuda_dp4a(qm[r][m][2], a.z, dot);
                        dot = ggml_cuda_dp4a(qm[r][m][3], a.w, dot);
                        tot[k][r] += d8 * (float) (scm[r][m] * (dot - 32 * sm));
                    }
                }
            }
#pragma unroll
            for (int k = 0; k < N; ++k) {
#pragma unroll
                for (int r = 0; r < R; ++r) acc[k][r] += dw[r] * tot[k][r];
            }
        }
    }
#pragma unroll
    for (int k = 0; k < N; ++k) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[k][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) dst[(size_t) k * stride_col_dst + row0 + r] = v;
        }
    }
}

template <int N>
static void mul_mat_vec_q6_K_pascal_mc_launch(const q6k_shadow_t & sh, const int * y, float * dst, int ncols, int nrows,
        int stride_col_y, int stride_col_dst, int W, int R, cudaStream_t stream) {
    const int nsb = ncols / 256;
    const int csb = std::max(1, std::min(nsb, 46000 / (384 * N)));
    const size_t smem = (size_t) N * csb * 384;
    const half * dh = (const half *) sh.d;
#define Q6MC(WW, RR) mul_mat_vec_q6_K_pascal_mc<WW, RR, N><<<dim3(nrows / (WW * RR)), dim3(32, WW), smem, stream>>>( \
        sh.ql, sh.qh, sh.sc, dh, y, dst, ncols, csb, stride_col_y, stride_col_dst)
    if (W == 4 && R == 2)      { Q6MC(4, 2); }
    else if (W == 8 && R == 1) { Q6MC(8, 1); }
    else if (W == 8 && R == 2) { Q6MC(8, 2); }
    else if (W == 8 && R == 4) { Q6MC(8, 4); }
    else if (W == 16 && R == 1) { Q6MC(16, 1); }
    else if (W == 16 && R == 2) { Q6MC(16, 2); }
    else                       { Q6MC(4, 1); }
#undef Q6MC
}

static bool mul_mat_vec_q6_K_pascal_mc_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x, const int stride_col_y,
        const int stride_col_dst, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_MULTICOL"); return !e || atoi(e) != 0; }();
    static const bool enabled6 = [] { const char * e = getenv("GGML_CUDA_Q6K_PASCAL"); return !e || atoi(e) != 0; }();
    if (!enabled || !enabled6 || ids || ncols_dst < 2 || ncols_dst > 8 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 256 != 0 || stride_row_x != ncols_x / 256) return false;
    if (fusion.gate || fusion.x_bias || fusion.shared_up || fusion.x_scale || fusion.gate_scale) return false;
    static const int cfg = [] { const char * e = getenv("GGML_CUDA_MC6_CFG"); return e ? atoi(e) : 2; }();
    int W = 8, R = 1;
    if (cfg == 1) { W = 4; R = 2; } else if (cfg == 2) { W = 8; R = 2; } else if (cfg == 3) { W = 4; R = 1; } else if (cfg == 4) { W = 8; R = 4; } else if (cfg == 5) { W = 16; R = 1; } else if (cfg == 6) { W = 16; R = 2; }
    if (nrows_x % (W * R) != 0) return false;
    q6k_shadow_t sh;
    if (!q6k_get_shadow(vx, (size_t) nrows_x * (ncols_x / 256), stream, sh)) return false;
    const int * y = (const int *) vy;
    switch (ncols_dst) {
        case 2: mul_mat_vec_q6_K_pascal_mc_launch<2>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 3: mul_mat_vec_q6_K_pascal_mc_launch<3>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 4: mul_mat_vec_q6_K_pascal_mc_launch<4>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 5: mul_mat_vec_q6_K_pascal_mc_launch<5>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 6: mul_mat_vec_q6_K_pascal_mc_launch<6>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        case 7: mul_mat_vec_q6_K_pascal_mc_launch<7>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
        default: mul_mat_vec_q6_K_pascal_mc_launch<8>(sh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, W, R, stream); break;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Pascal Q8_0 matvec (ncols_dst == 1). Q8_0 blocks are 34 B (not 16 B aligned), so each tensor is copied once
// into one aligned shadow buffer: qs stream (32 B per block) then d stream (2 B per block).
// ---------------------------------------------------------------------------------------------
static std::map<std::pair<const void *, size_t>, uint8_t *> & q80_shadows() {
    static std::map<std::pair<const void *, size_t>, uint8_t *> m;
    return m;
}
static std::mutex q80_mtx;

void ggml_cuda_q80_shadow_release(const void * base, size_t size) {
    std::lock_guard<std::mutex> lock(q80_mtx);
    auto & map = q80_shadows();
    bool synced = false;
    for (auto it = map.begin(); it != map.end();) {
        const char * p = (const char *) it->first.first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            if (!synced) { cudaDeviceSynchronize(); synced = true; }
            if (it->second) { cudaFree(it->second); }
            it = map.erase(it);
        } else {
            ++it;
        }
    }
}

static __global__ void q80_repack_kernel(const uint8_t * __restrict__ src, uint8_t * __restrict__ qs, __half * __restrict__ d, size_t nblocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    const uint8_t * s = src + b * 34;
    d[b] = __ushort_as_half((unsigned short) (s[0] | (s[1] << 8)));
    uint8_t * q = qs + b * 32;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[i] = s[2 + i];
}

static uint8_t * q80_get_shadow(const void * vx, const size_t nblocks, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(q80_mtx);
    auto & map = q80_shadows();
    auto it = map.find({vx, nblocks});
    static const bool nocache = [] { const char * e = getenv("GGML_CUDA_Q6K_NOCACHE"); return e && atoi(e) != 0; }();
    if (nocache && it != map.end()) {
        cudaDeviceSynchronize();
        if (it->second) { cudaFree(it->second); }
        map.erase(it);
        it = map.end();
    }
    if (it == map.end()) {
        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) {
            cudaGetLastError();
            return nullptr;
        }
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t need = nblocks * 34;
        uint8_t * buf = nullptr;
        if (free_b < need + (size_t) 768 * 1024 * 1024 || cudaMalloc(&buf, need) != cudaSuccess) {
            cudaGetLastError();
            map[{vx, nblocks}] = nullptr;
            return nullptr;
        }
        q80_repack_kernel<<<(unsigned) ((nblocks + 127) / 128), 128, 0, stream>>>((const uint8_t *) vx, buf, (__half *) (buf + nblocks * 32), nblocks);
        CUDA_CHECK(cudaGetLastError());
        it = map.emplace(std::make_pair(vx, nblocks), buf).first;
    }
    return it->second;
}

// Q8O: the block also writes the Q8_1 copy of its W*R == 32 output rows (the input of the next mat-vec) to q8o.
template <int W, int R, bool GATE, bool Q8O = false>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q8_0_pascal(const uint8_t * __restrict__ QS, const __half * __restrict__ D,
        const uint8_t * __restrict__ QSg, const __half * __restrict__ Dg, const int * __restrict__ y, float * __restrict__ out,
        const float * __restrict__ x_bias, const float * __restrict__ gate_bias, const ggml_glu_op glu_op, const float glu_limit, const int K,
        block_q8_1 * __restrict__ q8o) {
    constexpr int NG = GATE ? 2 : 1;
    static_assert(!Q8O || W * R == QK8_1, "Q8_1 output needs 32 rows per block");
    extern __shared__ int4 q80_smem4[];
    const int nblk = K >> 5;
    int4 *  yq = q80_smem4;
    float * yd = (float *) (q80_smem4 + nblk * 2);
    float * rbuf = yd + nblk;
    const int tix = threadIdx.x + threadIdx.y * 32;
    for (int i = tix; i < nblk * 9; i += W * 32) {
        const int b = i / 9, j = i - b * 9;
        if (j == 0) yd[b] = __low2float(*(const __half2 *) &y[i]);
        else ((int *) yq)[b * 8 + j - 1] = y[i];
    }
    __syncthreads();
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int h = lane & 1, b0 = lane >> 1;
    const int row0 = (blockIdx.x * W + warp) * R;
    float acc[NG][R];
#pragma unroll
    for (int t = 0; t < NG; ++t) {
#pragma unroll
        for (int r = 0; r < R; ++r) acc[t][r] = 0.f;
    }
    for (int blk = b0; blk < nblk; blk += 16) {
        uint4 w[NG][R]; __half dh[NG][R];
#pragma unroll
        for (int t = 0; t < NG; ++t) {
            const uint8_t * qs = t == 0 ? QS : QSg;
            const __half *  dd = t == 0 ? D : Dg;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const size_t rb = (size_t) (row0 + r) * nblk + blk;
                w[t][r]  = __ldg((const uint4 *) (qs + rb * 32) + h);
                dh[t][r] = dd[rb];
            }
        }
        const int4  a  = yq[blk * 2 + h];
        const float dy = yd[blk];
#pragma unroll
        for (int t = 0; t < NG; ++t) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s = __dp4a((int) w[t][r].x, a.x, 0);
                s = __dp4a((int) w[t][r].y, a.y, s);
                s = __dp4a((int) w[t][r].z, a.z, s);
                s = __dp4a((int) w[t][r].w, a.w, s);
                acc[t][r] += __half2float(dh[t][r]) * dy * (float) s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NG; ++t) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[t][r];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            acc[t][r] = v;
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = row0 + r;
            float result = acc[0][r];
            if (x_bias) result += x_bias[row];
            if constexpr (GATE) {
                float gv = acc[1][r];
                if (gate_bias) gv += gate_bias[row];
                switch (glu_op) {
                    case GGML_GLU_OP_SWIGLU:       result *= ggml_cuda_op_silu_single(gv); break;
                    case GGML_GLU_OP_GEGLU:        result *= ggml_cuda_op_gelu_single(gv); break;
                    case GGML_GLU_OP_SWIGLU_OAI:   result = ggml_cuda_op_swiglu_oai_single(gv, result); break;
                    case GGML_GLU_OP_SWIGLU_CLAMP: result = ggml_cuda_op_swiglu_clamp_single(gv, result, glu_limit); break;
                    default:                       result = result * gv; break;
                }
            }
            out[row] = result;
            if constexpr (Q8O) rbuf[warp * R + r] = result;
        }
    }
    if constexpr (Q8O) {
        __syncthreads();
        if (warp == 0) {
            const float  v    = rbuf[lane];
            const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
            const float  d    = amax / 127.0f;
            const int8_t q    = amax == 0.0f ? 0 : roundf(v / d);
            const int    sum  = warp_reduce_sum<QK8_1>((int) q);
            block_q8_1 * b    = q8o + blockIdx.x;
            b->qs[lane] = q;
            if (lane == 0) b->ds = make_half2(d, d * sum);
        }
    }
}

// Armed by ggml_cuda_mul_mat_vec_q: the fused gate/up kernel then also emits the Q8_1 copy of its output
static block_q8_1 * g_glu_q8_request = nullptr;
static bool         g_glu_q8_done    = false;

static bool mul_mat_vec_q8_0_pascal_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_Q80_PASCAL"); return !e || atoi(e) != 0; }();
    if (!enabled || ids || ncols_dst != 1 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 32 != 0 || stride_row_x != ncols_x / 32) return false;
    if (fusion.shared_up || fusion.x_scale || fusion.gate_scale) return false;
    const size_t smem = (size_t) (ncols_x / 32) * 36;
    if (smem > 48 * 1024) return false;
    const bool gate = fusion.gate != nullptr;
    int W, R;
    if (gate) {
        if (nrows_x % 8 != 0) return false;
        W = 8; R = 1;
    } else if (nrows_x % 16 == 0) { W = 8; R = 2; }
    else if (nrows_x % 8 == 0)    { W = 4; R = 2; }
    else if (nrows_x % 4 == 0)    { W = 4; R = 1; }
    else return false;

    const size_t nblocks = (size_t) nrows_x * (ncols_x / 32);
    uint8_t * sh = q80_get_shadow(vx, nblocks, stream);
    if (!sh) return false;
    uint8_t * shg = nullptr;
    if (gate) {
        shg = q80_get_shadow(fusion.gate, nblocks, stream);
        if (!shg) return false;
    }
    const __half * dh  = (const __half *) (sh + nblocks * 32);
    const __half * dhg = gate ? (const __half *) (shg + nblocks * 32) : nullptr;
    const float *  xb  = (const float *) fusion.x_bias;
    const float *  gb  = gate ? (const float *) fusion.gate_bias : nullptr;
    const dim3 block(32, W);
    const int * y = (const int *) vy;
    if (gate && g_glu_q8_request && nrows_x % QK8_1 == 0) {
        mul_mat_vec_q8_0_pascal<8, 4, true, true><<<nrows_x / 32, block, smem + 32 * sizeof(float), stream>>>(sh, dh, shg, dhg, y, dst, xb, gb, fusion.glu_op, fusion.glu_limit, ncols_x, g_glu_q8_request);
        g_glu_q8_done = true;
        return true;
    }
    if (gate)        mul_mat_vec_q8_0_pascal<8, 1, true ><<<nrows_x / 8,  block, smem, stream>>>(sh, dh, shg, dhg, y, dst, xb, gb, fusion.glu_op, fusion.glu_limit, ncols_x, nullptr);
    else if (W == 8) mul_mat_vec_q8_0_pascal<8, 2, false><<<nrows_x / 16, block, smem, stream>>>(sh, dh, nullptr, nullptr, y, dst, xb, nullptr, fusion.glu_op, fusion.glu_limit, ncols_x, nullptr);
    else if (R == 2) mul_mat_vec_q8_0_pascal<4, 2, false><<<nrows_x / 8,  block, smem, stream>>>(sh, dh, nullptr, nullptr, y, dst, xb, nullptr, fusion.glu_op, fusion.glu_limit, ncols_x, nullptr);
    else             mul_mat_vec_q8_0_pascal<4, 1, false><<<nrows_x / 4,  block, smem, stream>>>(sh, dh, nullptr, nullptr, y, dst, xb, nullptr, fusion.glu_op, fusion.glu_limit, ncols_x, nullptr);
    return true;
}

// Multi-column (2..8) Q8_0 matvec on the shadow weights: each weight is loaded once and used for every column.
template <int W, int R, int N>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q8_0_pascal_mc(const uint8_t * __restrict__ QS, const __half * __restrict__ D, const int * __restrict__ y,
        float * __restrict__ out, const int K, const int stride_y, const int stride_dst) {
    extern __shared__ int4 q80mc_smem4[];
    const int nblk = K >> 5;
    int4 *  yq = q80mc_smem4;                          // N * nblk * 2 int4
    float * yd = (float *) (q80mc_smem4 + N * nblk * 2); // N * nblk
    const int tix = threadIdx.x + threadIdx.y * 32;
    for (int i = tix; i < N * nblk * 9; i += W * 32) {
        const int c = i / (nblk * 9), ii = i - c * (nblk * 9);
        const int b = ii / 9, j = ii - b * 9;
        const int v = y[(size_t) c * stride_y + ii];
        if (j == 0) yd[c * nblk + b] = __low2float(*(const __half2 *) &v);
        else ((int *) yq)[(c * nblk + b) * 8 + j - 1] = v;
    }
    __syncthreads();
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int h = lane & 1, b0 = lane >> 1;
    const int row0 = (blockIdx.x * W + warp) * R;
    float acc[R][N];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < N; ++c) acc[r][c] = 0.f;
    }
    for (int blk = b0; blk < nblk; blk += 16) {
        uint4 w[R]; float dw[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t rb = (size_t) (row0 + r) * nblk + blk;
            w[r]  = __ldg((const uint4 *) (QS + rb * 32) + h);
            dw[r] = __half2float(D[rb]);
        }
#pragma unroll
        for (int c = 0; c < N; ++c) {
            const int4  a  = yq[(c * nblk + blk) * 2 + h];
            const float dy = yd[c * nblk + blk];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int s = __dp4a((int) w[r].x, a.x, 0);
                s = __dp4a((int) w[r].y, a.y, s);
                s = __dp4a((int) w[r].z, a.z, s);
                s = __dp4a((int) w[r].w, a.w, s);
                acc[r][c] += dw[r] * dy * (float) s;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < N; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (lane == 0) out[(size_t) c * stride_dst + row0 + r] = v;
        }
    }
}

template <int N>
static void mul_mat_vec_q8_0_pascal_mc_launch(const uint8_t * sh, const __half * dh, const int * y, float * dst, int K, int nrows,
        int stride_y, int stride_dst, size_t smem, cudaStream_t stream) {
    mul_mat_vec_q8_0_pascal_mc<8, 2, N><<<nrows / 16, dim3(32, 8), smem, stream>>>(sh, dh, y, dst, K, stride_y, stride_dst);
}

static bool mul_mat_vec_q8_0_pascal_mc_try(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst, const int stride_row_x,
        const int stride_col_y, const int stride_col_dst, const int nchannels_dst, const int nsamples_dst, cudaStream_t stream) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_Q80_PASCAL"); const char * m = getenv("GGML_CUDA_Q80_MULTICOL");
        return (!e || atoi(e) != 0) && m && atoi(m) != 0;
    }();
    if (!enabled || ids || ncols_dst < 2 || ncols_dst > 8 || nchannels_dst != 1 || nsamples_dst != 1) return false;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA)) return false;
    if (ncols_x % 32 != 0 || stride_row_x != ncols_x / 32 || nrows_x % 16 != 0) return false;
    if (fusion.gate || fusion.x_bias || fusion.gate_bias || fusion.shared_up || fusion.x_scale || fusion.gate_scale) return false;
    if (stride_col_y != (ncols_x / 32) * 9) return false;
    const size_t smem = (size_t) ncols_dst * (ncols_x / 32) * 36;
    if (smem > 46000) return false;
    const size_t nblocks = (size_t) nrows_x * (ncols_x / 32);
    uint8_t * sh = q80_get_shadow(vx, nblocks, stream);
    if (!sh) return false;
    const __half * dh = (const __half *) (sh + nblocks * 32);
    const int * y = (const int *) vy;
    switch (ncols_dst) {
        case 2: mul_mat_vec_q8_0_pascal_mc_launch<2>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        case 3: mul_mat_vec_q8_0_pascal_mc_launch<3>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        case 4: mul_mat_vec_q8_0_pascal_mc_launch<4>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        case 5: mul_mat_vec_q8_0_pascal_mc_launch<5>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        case 6: mul_mat_vec_q8_0_pascal_mc_launch<6>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        case 7: mul_mat_vec_q8_0_pascal_mc_launch<7>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
        default: mul_mat_vec_q8_0_pascal_mc_launch<8>(sh, dh, y, dst, ncols_x, nrows_x, stride_col_y, stride_col_dst, smem, stream); break;
    }
    return true;
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    if constexpr (type == GGML_TYPE_Q8_0) {
        if (mul_mat_vec_q8_0_pascal_mc_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                           stride_col_y, stride_col_dst, nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q8_0) {
        if (mul_mat_vec_q8_0_pascal_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                        nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q6_K) {
        if (mul_mat_vec_q6_K_pascal_mc_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                           stride_col_y, stride_col_dst, nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q6_K) {
        if (mul_mat_vec_q6_K_pascal_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                        nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (mul_mat_vec_q4_K_pascal_mc_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                           stride_col_y, stride_col_dst, nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }
    if constexpr (type == GGML_TYPE_Q4_K) {
        if (mul_mat_vec_q4_K_pascal_try(vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x,
                                        nchannels_dst, nsamples_dst, stream)) {
            return;
        }
    }

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst + (fusion.shared_up != nullptr),
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            if (should_use_small_k(c_ncols_dst)) {
                launch(std::true_type{},  std::false_type{});
            } else if (should_halve_iters()) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream) {
    switch (type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// One-entry cache of the last Q8_1 quantisation of src1 (per device). Valid for one graph evaluation only.
#define QCACHE_CAP ((size_t) 2 * 1024 * 1024)
struct qcache_t {
    const ggml_tensor * tensor = nullptr;
    const void *        data   = nullptr;
    cudaStream_t        stream = nullptr;
    uint64_t            epoch  = 0;
    int64_t             ne[4]  = {0, 0, 0, 0};
    size_t              nb[4]  = {0, 0, 0, 0};
    char *              buf[2] = {nullptr, nullptr}; // two buffers: a writer never overwrites the copy a reader still uses
    int                 cur    = 0;                  // buffer holding the published copy
};
static qcache_t g_qcache[GGML_CUDA_MAX_DEVICES];
static uint64_t g_qcache_epoch = 1;

static bool qcache_ensure(qcache_t & qc, cudaStream_t stream) {
    if (qc.buf[0] && qc.buf[1]) return true;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &st) != cudaSuccess || st != cudaStreamCaptureStatusNone) { cudaGetLastError(); return false; }
    for (int k = 0; k < 2; ++k) {
        if (!qc.buf[k] && cudaMalloc((void **) &qc.buf[k], QCACHE_CAP) != cudaSuccess) { cudaGetLastError(); qc.buf[k] = nullptr; return false; }
    }
    return true;
}
static char * qcache_wbuf(qcache_t & qc) { return qc.buf[1 - qc.cur]; }
static void qcache_publish(qcache_t & qc, const ggml_tensor * t, cudaStream_t stream) {
    qc.cur = 1 - qc.cur;
    qc.tensor = t; qc.data = t->data; qc.stream = stream; qc.epoch = g_qcache_epoch;
    for (int i = 0; i < 4; ++i) { qc.ne[i] = t->ne[i]; qc.nb[i] = t->nb[i]; }
}

void ggml_cuda_qcache_new_epoch() {
    g_qcache_epoch++;
}

// RMS norm + mul that also writes the Q8_1 copy of its output into the quantisation cache (Pascal only).
// Same block size and reduction order as the stock rms_norm_f32 vector path, so the f32 output is unchanged.
// ADD: the input is the sum x + xb (residual add), which is also written to add_out (it stays live as the next residual).
template <int BS, int VPT, bool ADD = false>
static __global__ void __launch_bounds__(BS)
rms_norm_mul_q8_1_kernel(const float * __restrict__ x, const float * __restrict__ mul, float * __restrict__ dst,
        block_q8_1 * __restrict__ q8, const int ncols, const int64_t stride_row, const float eps,
        const float * __restrict__ xb = nullptr, float * __restrict__ add_out = nullptr) {
    extern __shared__ float s_sum[];
    const int row = blockIdx.x, tid = threadIdx.x;
    const float * xc   = x + row * stride_row;
    float *       dstc = dst + (size_t) row * ncols;
    const int ncols4 = ncols / 4;
    float4 xs[VPT];
    float  tmp = 0.0f;
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int i = tid + k * BS;
        if (i < ncols4) {
            xs[k] = ((const float4 *) xc)[i];
            if constexpr (ADD) {
                const float4 b = ((const float4 *) (xb + row * stride_row))[i];
                xs[k] = make_float4(xs[k].x + b.x, xs[k].y + b.y, xs[k].z + b.z, xs[k].w + b.w);
                ((float4 *) (add_out + row * stride_row))[i] = xs[k];
            }
            tmp += xs[k].x*xs[k].x + xs[k].y*xs[k].y + xs[k].z*xs[k].z + xs[k].w*xs[k].w;
        }
    }
    tmp = block_reduce<block_reduce_method::SUM, BS>(tmp, s_sum);
    const float scale = rsqrtf(tmp / ncols + eps);
    block_q8_1 * q8r = q8 + (size_t) row * (ncols / QK8_1);
#pragma unroll
    for (int k = 0; k < VPT; k++) {
        const int  i  = tid + k * BS;
        const bool in = i < ncols4;
        float4 r = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (in) {
            const float4 m = ((const float4 *) mul)[i];
            r = make_float4(scale*xs[k].x, scale*xs[k].y, scale*xs[k].z, scale*xs[k].w);
            r.x *= m.x; r.y *= m.y; r.z *= m.z; r.w *= m.w;
            ((float4 *) dstc)[i] = r;
        }
        float amax = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float d = amax / 127.0f;
        const int8_t q0 = amax == 0.0f ? 0 : (int8_t) roundf(r.x / d);
        const int8_t q1 = amax == 0.0f ? 0 : (int8_t) roundf(r.y / d);
        const int8_t q2 = amax == 0.0f ? 0 : (int8_t) roundf(r.z / d);
        const int8_t q3 = amax == 0.0f ? 0 : (int8_t) roundf(r.w / d);
        int sum = (int) q0 + (int) q1 + (int) q2 + (int) q3;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (in) {
            const uint32_t packed = (uint32_t) (uint8_t) q0 | ((uint32_t) (uint8_t) q1 << 8) | ((uint32_t) (uint8_t) q2 << 16) | ((uint32_t) (uint8_t) q3 << 24);
            *(uint32_t *) (q8r[i >> 3].qs + (i & 7) * 4) = packed;
            if ((i & 7) == 0) q8r[i >> 3].ds = make_half2(d, d * sum);
        }
    }
}

static bool ggml_cuda_tensor_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_nbytes(a);
    const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// add_t != nullptr: the node before rms is the residual ADD feeding it; it is executed by the same kernel.
static bool ggml_cuda_rms_norm_mul_q8_1_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * add_t, const ggml_tensor * rms, const ggml_tensor * mul_t) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_NORM_Q8"); return !e || atoi(e) != 0; }();
    if (!enabled) return false;
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA) || dev >= GGML_CUDA_MAX_DEVICES) return false;
    const ggml_tensor * x = add_t ? add_t->src[0] : rms->src[0];
    const ggml_tensor * xb = add_t ? add_t->src[1] : nullptr;
    const ggml_tensor * m = mul_t->src[0] == rms ? mul_t->src[1] : mul_t->src[0];
    if (add_t) {
        if (add_t->type != GGML_TYPE_F32 || xb->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, xb) || !ggml_are_same_shape(x, add_t)) return false;
        if (!ggml_is_contiguous(x) || !ggml_is_contiguous(xb) || !ggml_is_contiguous(add_t)) return false;
        if (((uintptr_t) xb->data & 15) || ((uintptr_t) add_t->data & 15)) return false;
        // in-place add (output == one input) is fine element-wise; any other overlap is not
        for (const ggml_tensor * in : { x, xb }) {
            if (ggml_cuda_tensor_ranges_overlap(add_t, in) && add_t->data != in->data) return false;
        }
        if (ggml_cuda_tensor_ranges_overlap(add_t, m) || ggml_cuda_tensor_ranges_overlap(add_t, mul_t)) return false;
        if (ggml_cuda_tensor_ranges_overlap(mul_t, x) || ggml_cuda_tensor_ranges_overlap(mul_t, xb) || ggml_cuda_tensor_ranges_overlap(mul_t, m)) return false;
    }
    if (x->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || mul_t->type != GGML_TYPE_F32) return false;
    const int64_t ncols = x->ne[0], nrows = x->ne[1];
    if (ncols % 512 != 0 || nrows < 1 || nrows > 8 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (x->nb[0] != 4 || x->nb[1] % 16 != 0 || ((uintptr_t) x->data & 15)) return false;
    if (m->nb[0] != 4 || m->ne[0] != ncols || m->ne[1] != 1 || m->ne[2] != 1 || m->ne[3] != 1 || ((uintptr_t) m->data & 15)) return false;
    if (!ggml_is_contiguous(mul_t) || mul_t->ne[0] != ncols || mul_t->ne[1] != nrows || ((uintptr_t) mul_t->data & 15)) return false;
    const size_t q8_bytes = (size_t) nrows * (ncols / QK8_1) * sizeof(block_q8_1);
    if (q8_bytes > QCACHE_CAP) return false;
    cudaStream_t stream = ctx.stream();
    qcache_t & qc = g_qcache[dev];
    if (!qcache_ensure(qc, stream)) return false;
    float eps = 0.0f;
    memcpy(&eps, rms->op_params, sizeof(float));
    const int n4 = (int) (ncols / 4);
    const int64_t s01 = x->nb[1] / 4;
    const float * xd = (const float *) x->data;
    const float * md = (const float *) m->data;
    float *       dd = (float *) mul_t->data;
    block_q8_1 *  q8 = (block_q8_1 *) qcache_wbuf(qc);
    if (add_t) {
        const float * xbd = (const float *) xb->data;
        float *       ado = (float *) add_t->data;
        if (n4 <= 256)       rms_norm_mul_q8_1_kernel<256, 1, true><<<nrows, 256, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps, xbd, ado);
        else if (n4 <= 512)  rms_norm_mul_q8_1_kernel<512, 1, true><<<nrows, 512, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps, xbd, ado);
        else if (n4 <= 1024) rms_norm_mul_q8_1_kernel<1024, 1, true><<<nrows, 1024, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps, xbd, ado);
        else if (n4 <= 4096) rms_norm_mul_q8_1_kernel<1024, 4, true><<<nrows, 1024, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps, xbd, ado);
        else return false;
    } else {
        if (n4 <= 256)       rms_norm_mul_q8_1_kernel<256, 1><<<nrows, 256, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps);
        else if (n4 <= 512)  rms_norm_mul_q8_1_kernel<512, 1><<<nrows, 512, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps);
        else if (n4 <= 1024) rms_norm_mul_q8_1_kernel<1024, 1><<<nrows, 1024, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps);
        else if (n4 <= 4096) rms_norm_mul_q8_1_kernel<1024, 4><<<nrows, 1024, 32 * sizeof(float), stream>>>(xd, md, dd, q8, (int) ncols, s01, eps);
        else return false;
    }
    CUDA_CHECK(cudaGetLastError());
    qcache_publish(qc, mul_t, stream);
    return true;
}

bool ggml_cuda_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * rms, const ggml_tensor * mul_t) {
    return ggml_cuda_rms_norm_mul_q8_1_impl(ctx, nullptr, rms, mul_t);
}

bool ggml_cuda_add_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, const ggml_tensor * add_t, const ggml_tensor * rms, const ggml_tensor * mul_t) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_ADD_NORM"); return !e || atoi(e) != 0; }();
    return enabled && ggml_cuda_rms_norm_mul_q8_1_impl(ctx, add_t, rms, mul_t);
}

// SwiGLU (silu(x) * g) that also writes the Q8_1 copy of its output for the following mat-vec (Pascal only).
// One thread per element, one warp per Q8_1 block; same arithmetic as unary_gated_op_kernel<op_silu>.
static __global__ void __launch_bounds__(256)
swiglu_q8_1_kernel(const float * __restrict__ x, const float * __restrict__ g, float * __restrict__ dst, block_q8_1 * __restrict__ q8,
        const int64_t nc, const int64_t o0, const int64_t o1) {
    const int64_t i   = (int64_t) blockIdx.x * 256 + threadIdx.x;
    const int64_t row = i / nc, col = i - row * nc;
    const float v = ggml_cuda_op_silu_single(x[row * o0 + col]) * g[row * o1 + col];
    dst[i] = v;
    const float  amax = warp_reduce_max<QK8_1>(fabsf(v));
    const float  d    = amax / 127.0f;
    const int8_t q    = amax == 0.0f ? 0 : roundf(v / d);
    const int    sum  = warp_reduce_sum<QK8_1>((int) q);
    block_q8_1 * b = q8 + (i >> 5);
    b->qs[i & 31] = q;
    if ((i & 31) == 0) b->ds = make_half2(d, d * sum);
}

bool ggml_cuda_glu_q8_1(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_GLU2_Q8"); return !e || atoi(e) != 0; }();
    if (!enabled) return false;
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA) || dev >= GGML_CUDA_MAX_DEVICES) return false;
    const ggml_tensor * node = cgraph->nodes[i];
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (node->op != GGML_OP_GLU || ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU || !a || !b) return false;
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    const int64_t nc = node->ne[0], nrows = node->ne[1];
    if (nc % 512 != 0 || nrows < 1 || nrows > 8 || node->ne[2] != 1 || node->ne[3] != 1 || !ggml_is_contiguous(node)) return false;
    if (a->ne[0] != nc || b->ne[0] != nc || a->ne[1] != nrows || b->ne[1] != nrows || a->ne[2] != 1 || b->ne[2] != 1 || a->ne[3] != 1 || b->ne[3] != 1) return false;
    if (a->nb[0] != 4 || b->nb[0] != 4 || a->nb[1] % 4 != 0 || b->nb[1] % 4 != 0) return false;
    if ((size_t) nrows * (nc / QK8_1) * sizeof(block_q8_1) > QCACHE_CAP) return false;
    bool consumer = false;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 4 && !consumer; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        consumer = c->op == GGML_OP_MUL_MAT && c->src[1] == node &&
                   (c->src[0]->type == GGML_TYPE_Q8_0 || c->src[0]->type == GGML_TYPE_Q4_K || c->src[0]->type == GGML_TYPE_Q6_K);
    }
    if (!consumer) return false;
    cudaStream_t stream = ctx.stream();
    qcache_t & qc = g_qcache[dev];
    if (!qcache_ensure(qc, stream)) return false;
    const int64_t k = nc * nrows;
    swiglu_q8_1_kernel<<<(unsigned) (k / 256), 256, 0, stream>>>((const float *) a->data, (const float *) b->data, (float *) node->data,
        (block_q8_1 *) qcache_wbuf(qc), nc, a->nb[1] / 4, b->nb[1] / 4);
    CUDA_CHECK(cudaGetLastError());
    qcache_publish(qc, node, stream);
    return true;
}

// Flash attention output -> Q8_1 for the attention output projection (Pascal only).
block_q8_1 * g_fa_q8_request = nullptr;
bool         g_fa_q8_done    = false;
static const ggml_tensor * g_fa_q8_x = nullptr;

void ggml_cuda_fa_q8_prepare(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    g_fa_q8_request = nullptr;
    g_fa_q8_done    = false;
    g_fa_q8_x       = nullptr;
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_FA_Q8"); return !e || atoi(e) != 0; }();
    if (!enabled) return;
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA) || dev >= GGML_CUDA_MAX_DEVICES) return;
    const ggml_tensor * fa = cgraph->nodes[i];
    if (fa->type != GGML_TYPE_F32 || fa->ne[0] % QK8_1 != 0 || fa->ne[3] != 1 || !ggml_is_contiguous(fa)) return;
    for (int j = i + 1; j < cgraph->n_nodes && j < i + 6; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_MUL_MAT || !(c->src[0]->type == GGML_TYPE_Q8_0 || c->src[0]->type == GGML_TYPE_Q4_K || c->src[0]->type == GGML_TYPE_Q6_K)) continue;
        const ggml_tensor * x = c->src[1];
        const bool same = x == fa || ((x->op == GGML_OP_RESHAPE || x->op == GGML_OP_VIEW) && x->src[0] == fa && x->data == fa->data);
        if (!same) continue;
        const int64_t ncols = x->ne[0], nrows = x->ne[1];
        if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || ncols != fa->ne[0] * fa->ne[1] || nrows != fa->ne[2] || x->ne[2] != 1 || x->ne[3] != 1) return;
        if (ncols % 512 != 0 || nrows < 1 || nrows > 8) return;
        const size_t q8_bytes = (size_t) nrows * (ncols / QK8_1) * sizeof(block_q8_1);
        if (q8_bytes > QCACHE_CAP) return;
        cudaStream_t stream = ctx.stream();
        qcache_t & qc = g_qcache[dev];
        if (!qcache_ensure(qc, stream)) return;
        g_fa_q8_request = (block_q8_1 *) qcache_wbuf(qc);
        g_fa_q8_x       = x;
        return;
    }
}

void ggml_cuda_fa_q8_finish(ggml_backend_cuda_context & ctx) {
    const ggml_tensor * x = g_fa_q8_x;
    const bool done = g_fa_q8_done;
    g_fa_q8_request = nullptr;
    g_fa_q8_done    = false;
    g_fa_q8_x       = nullptr;
    if (!done || !x) return;
    const int dev = ggml_cuda_get_device();
    qcache_publish(g_qcache[dev], x, ctx.stream());
}

// Three Q8_0 mat-vecs reading the same input (q, k, v projections) in one launch (Pascal, 1 column).
struct q80_multi_args {
    const uint8_t * qs[3];
    const __half *  d[3];
    float *         out[3];
    int             nb[3];   // blocks (of W*R rows) per matrix
    // optional: write the result as f16 into row idx[0] of a KV cache (fused SET_ROWS) instead of out[]
    __half *        cache[3];
    const int64_t * cidx[3];
    long long       cstride[3];
};

template <int W, int R>
static __global__ void __launch_bounds__(W * 32)
mul_mat_vec_q8_0_pascal_multi(const q80_multi_args a, const int * __restrict__ y, const int K) {
    extern __shared__ int4 q80m_smem4[];
    const int nblk = K >> 5;
    int4 *  yq = q80m_smem4;
    float * yd = (float *) (q80m_smem4 + nblk * 2);
    int bx = blockIdx.x, m = 0;
    if (bx >= a.nb[0]) { bx -= a.nb[0]; m = 1; if (bx >= a.nb[1]) { bx -= a.nb[1]; m = 2; } }
    const uint8_t * QS  = m == 0 ? a.qs[0]  : m == 1 ? a.qs[1]  : a.qs[2];
    const __half *  D   = m == 0 ? a.d[0]   : m == 1 ? a.d[1]   : a.d[2];
    float *         out = m == 0 ? a.out[0] : m == 1 ? a.out[1] : a.out[2];
    __half *        cache = m == 0 ? a.cache[0] : m == 1 ? a.cache[1] : a.cache[2];
    if (cache) {
        const int64_t * cidx = m == 0 ? a.cidx[0] : m == 1 ? a.cidx[1] : a.cidx[2];
        const long long cstr = m == 0 ? a.cstride[0] : m == 1 ? a.cstride[1] : a.cstride[2];
        cache += cidx[0] * cstr;
    }
    const int tix = threadIdx.x + threadIdx.y * 32;
    for (int i = tix; i < nblk * 9; i += W * 32) {
        const int b = i / 9, j = i - b * 9;
        if (j == 0) yd[b] = __low2float(*(const __half2 *) &y[i]);
        else ((int *) yq)[b * 8 + j - 1] = y[i];
    }
    __syncthreads();
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int h = lane & 1, b0 = lane >> 1;
    const int row0 = (bx * W + warp) * R;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int blk = b0; blk < nblk; blk += 16) {
        uint4 w[R]; __half dh[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t rb = (size_t) (row0 + r) * nblk + blk;
            w[r]  = __ldg((const uint4 *) (QS + rb * 32) + h);
            dh[r] = D[rb];
        }
        const int4  a4 = yq[blk * 2 + h];
        const float dy = yd[blk];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int s = __dp4a((int) w[r].x, a4.x, 0);
            s = __dp4a((int) w[r].y, a4.y, s);
            s = __dp4a((int) w[r].z, a4.z, s);
            s = __dp4a((int) w[r].w, a4.w, s);
            acc[r] += __half2float(dh[r]) * dy * (float) s;
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) {
            if (cache) cache[row0 + r] = __float2half(v);
            else       out[row0 + r] = v;
        }
    }
}

// Nodes whose work was done by a fused kernel earlier in the same graph evaluation (e.g. the V SET_ROWS).
static std::vector<const ggml_tensor *> g_skip_nodes;
static uint64_t                         g_skip_epoch = 0;

bool ggml_cuda_node_skipped(const ggml_tensor * node) {
    if (g_skip_epoch != g_qcache_epoch || g_skip_nodes.empty()) return false;
    for (const ggml_tensor * t : g_skip_nodes) {
        if (t == node) return true;
    }
    return false;
}

void ggml_cuda_mark_node_skipped(const ggml_tensor * node) {
    if (g_skip_epoch != g_qcache_epoch) { g_skip_nodes.clear(); g_skip_epoch = g_qcache_epoch; }
    g_skip_nodes.push_back(node);
}

// For a mat-vec node: the SET_ROWS (into an F16 KV cache) that is its only consumer through RESHAPE/VIEW nodes, or nullptr.
const ggml_tensor * ggml_cuda_find_v_set_rows(const ggml_cgraph * cgraph, int from, const ggml_tensor * mm, int n_tokens) {
    const ggml_tensor * set_rows = nullptr;
    for (int j = from; j < cgraph->n_nodes && j < from + 40; ++j) {
        const ggml_tensor * c = cgraph->nodes[j];
        if (c->op != GGML_OP_SET_ROWS) continue;
        const ggml_tensor * s = c->src[0];
        int depth = 0;
        while (s && s != mm && (s->op == GGML_OP_VIEW || s->op == GGML_OP_RESHAPE) && depth++ < 4) s = s->src[0];
        if (s == mm) { set_rows = c; break; }
    }
    if (!set_rows || set_rows->type != GGML_TYPE_F16 || !set_rows->src[1] || set_rows->src[1]->type != GGML_TYPE_I64) return nullptr;
    if (set_rows->src[1]->ne[0] != n_tokens || set_rows->ne[0] != mm->ne[0] || ggml_nelements(set_rows->src[0]) != ggml_nelements(mm)) return nullptr;
    if (set_rows->nb[0] != sizeof(ggml_fp16_t) || set_rows->nb[1] % sizeof(ggml_fp16_t) != 0) return nullptr;
    // the matrix output must not be read by anything but the RESHAPE/VIEW/SET_ROWS chain
    for (int k = 0; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * u = cgraph->nodes[k];
        if (u == set_rows || u == mm) continue;
        for (int sidx = 0; sidx < GGML_MAX_SRC; ++sidx) {
            const ggml_tensor * s = u->src[sidx];
            if (!s) continue;
            const ggml_tensor * r = s;
            int depth = 0;
            while (r && r != mm && (r->op == GGML_OP_VIEW || r->op == GGML_OP_RESHAPE) && depth++ < 4) r = r->src[0];
            if (r == mm && u->op != GGML_OP_VIEW && u->op != GGML_OP_RESHAPE) return nullptr;
        }
    }
    return set_rows;
}

// Runs the mat-vecs of cgraph->nodes[i .. i+2] as one kernel when they are Q8_0, 1 column and share src1.
bool ggml_cuda_mul_mat_vec_q8_0_qkv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool enabled = [] {
        const char * e = getenv("GGML_CUDA_QKV_FUSE"); const char * p = getenv("GGML_CUDA_Q80_PASCAL");
        return (!e || atoi(e) != 0) && (!p || atoi(p) != 0);
    }();
    if (!enabled || i + 2 >= cgraph->n_nodes) return false;
    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA) || dev >= GGML_CUDA_MAX_DEVICES) return false;
    const ggml_tensor * n[3] = { cgraph->nodes[i], cgraph->nodes[i + 1], cgraph->nodes[i + 2] };
    const ggml_tensor * x = n[0]->src[1];
    const int64_t K = n[0]->src[0]->ne[0];
    for (int t = 0; t < 3; ++t) {
        const ggml_tensor * w = n[t]->src[0];
        if (n[t]->op != GGML_OP_MUL_MAT || n[t]->src[1] != x || w->type != GGML_TYPE_Q8_0 || w->ne[0] != K || w->ne[2] != 1 || w->ne[3] != 1) return false;
        if (n[t]->type != GGML_TYPE_F32 || !ggml_is_contiguous(n[t]) || n[t]->ne[1] != 1 || n[t]->ne[2] != 1 || n[t]->ne[3] != 1) return false;
        if (w->ne[1] % 16 != 0 || w->nb[1] != (size_t) (K / 32) * 34) return false;
    }
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != 4 || K % MATRIX_ROW_PADDING != 0) return false;
    const size_t smem = (size_t) (K / 32) * 36;
    if (smem > 48 * 1024 || (size_t) (K / 32) * sizeof(block_q8_1) > QCACHE_CAP) return false;
    cudaStream_t stream = ctx.stream();
    qcache_t & qc = g_qcache[dev];
    if (!qcache_ensure(qc, stream)) return false;

    q80_multi_args args{};
    int total_blocks = 0;
    for (int t = 0; t < 3; ++t) {
        const ggml_tensor * w = n[t]->src[0];
        const size_t nblocks = (size_t) w->ne[1] * (K / 32);
        uint8_t * sh = q80_get_shadow(w->data, nblocks, stream);
        if (!sh) return false;
        args.qs[t]  = sh;
        args.d[t]   = (const __half *) (sh + nblocks * 32);
        args.out[t] = (float *) n[t]->data;
        args.nb[t]  = (int) (w->ne[1] / 16);
        total_blocks += args.nb[t];
    }

    const bool hit = qc.tensor == x && qc.data == x->data && qc.stream == stream && qc.epoch == g_qcache_epoch &&
                     qc.ne[0] == x->ne[0] && qc.ne[1] == 1 && qc.ne[2] == 1 && qc.ne[3] == 1 && qc.nb[1] == x->nb[1];
    const char * q8 = hit ? qc.buf[qc.cur] : qcache_wbuf(qc);
    if (!hit) {
        const int64_t s11 = x->nb[1] / 4, s12 = x->nb[2] / 4, s13 = x->nb[3] / 4;
        quantize_row_q8_1_cuda((const float *) x->data, (void *) q8, K, s11, s12, s13, K, 1, 1, 1, stream);
        qcache_publish(qc, x, stream);
    }
    // V projection straight into the KV cache (skips the separate SET_ROWS launch)
    static const bool vcache_fuse = [] { const char * e = getenv("GGML_CUDA_V_CACHE_FUSE"); return !e || atoi(e) != 0; }();
    std::vector<const ggml_tensor *> fused_set_rows;
    if (vcache_fuse) {
        for (int t = 0; t < 3; ++t) {
            const ggml_tensor * sr = ggml_cuda_find_v_set_rows(cgraph, i + 3, n[t], 1);
            if (sr) {
                args.cache[t]   = (__half *) sr->data;
                args.cidx[t]    = (const int64_t *) sr->src[1]->data;
                args.cstride[t] = (long long) (sr->nb[1] / sizeof(ggml_fp16_t));
                fused_set_rows.push_back(sr);
            }
        }
    }

    mul_mat_vec_q8_0_pascal_multi<8, 2><<<total_blocks, dim3(32, 8), smem, stream>>>(args, (const int *) q8, (int) K);
    CUDA_CHECK(cudaGetLastError());
    if (!fused_set_rows.empty()) {
        if (g_skip_epoch != g_qcache_epoch) { g_skip_nodes.clear(); g_skip_epoch = g_qcache_epoch; }
        for (const ggml_tensor * sr : fused_set_rows) g_skip_nodes.push_back(sr);
    }
    return true;
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mm_fusion_args_device fusion_local{};

    if (fusion) {
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT( !ids || dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc));
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) || src0->type == GGML_TYPE_NVFP4);

        if (fusion->shared_up) {
            GGML_ASSERT(ids && fusion->gate && fusion->shared_gate && fusion->shared_dst);
            GGML_ASSERT(!fusion->x_bias && !fusion->gate_bias && !fusion->x_scale && !fusion->gate_scale);
            GGML_ASSERT(ne11 == 1 && ne03 == 1 && ne13 == 1);
            GGML_ASSERT(fusion->shared_up->type == src0->type && fusion->shared_gate->type == src0->type);
            GGML_ASSERT(ggml_are_same_shape(fusion->shared_up, fusion->shared_gate));
            GGML_ASSERT(ggml_is_contiguous(fusion->shared_up) && ggml_is_contiguous(fusion->shared_gate));
            GGML_ASSERT(fusion->shared_up->ne[0] == ne00 && fusion->shared_up->ne[1] == ne01);
            GGML_ASSERT(fusion->shared_up->nb[1] == nb01 && ggml_is_matrix(fusion->shared_up));
            GGML_ASSERT(fusion->shared_dst->type == GGML_TYPE_F32 && ggml_is_contiguous(fusion->shared_dst));
            GGML_ASSERT(fusion->shared_dst->ne[0] == ne0 && fusion->shared_dst->ne[1] == ne2);
            fusion_local.shared_up   = fusion->shared_up->data;
            fusion_local.shared_gate = fusion->shared_gate->data;
            fusion_local.shared_dst  = (float *) fusion->shared_dst->data;
            fusion_local.shared_stride_col_dst = fusion->shared_dst->nb[1] / ts_dst;
        }

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        fusion_local.glu_op = fusion->glu_op;
        fusion_local.glu_limit = fusion->glu_limit;
    }

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  q8_bytes    = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1;
    ggml_cuda_pool_alloc<char> src1_q8_1;
    char * q8_ptr = nullptr;
    bool   q8_hit = false;
    qcache_t * qc = nullptr;
    {
        // Pascal only: share the Q8_1 copy of src1 between consecutive mat-vecs reading the same tensor (q/k/v, gate/up)
        static const bool qc_enabled = [] { const char * e = getenv("GGML_CUDA_QCACHE"); return !e || atoi(e) != 0; }();
        const int dev = ggml_cuda_get_device();
        const int cc  = ggml_cuda_info().devices[dev].cc;
        if (qc_enabled && !ids && GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA && q8_bytes <= QCACHE_CAP && dev < GGML_CUDA_MAX_DEVICES) {
            qc = &g_qcache[dev];
            if (qcache_ensure(*qc, stream)) {
                q8_hit = qc->tensor == src1 && qc->data == src1->data && qc->stream == stream && qc->epoch == g_qcache_epoch &&
                         qc->ne[0] == ne10 && qc->ne[1] == ne11 && qc->ne[2] == ne12 && qc->ne[3] == ne13 &&
                         qc->nb[1] == src1->nb[1] && qc->nb[2] == src1->nb[2] && qc->nb[3] == src1->nb[3];
                q8_ptr = q8_hit ? qc->buf[qc->cur] : qcache_wbuf(*qc);
            } else {
                qc = nullptr;
            }
        }
        if (!q8_ptr) {
            q8_ptr = src1_q8_1.alloc(ctx.pool(), q8_bytes);
        }
        if (!q8_hit) {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            quantize_row_q8_1_cuda(src1_d, q8_ptr, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            if (qc) {
                qcache_publish(*qc, src1, stream);
            }
        }
    }

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    // Pascal: let the fused Q8_0 gate/up kernel also emit the Q8_1 input of the following (down) mat-vec
    static const bool glu_q8_enabled = [] { const char * e = getenv("GGML_CUDA_GLU_Q8"); return !e || atoi(e) != 0; }();
    qcache_t * qc_out = nullptr;
    if (glu_q8_enabled && fusion && fusion->gate && !ids && src0->type == GGML_TYPE_Q8_0 && ncols_dst == 1 && ne2 == 1 && ne3 == 1 &&
        dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) && dst->ne[0] % MATRIX_ROW_PADDING == 0) {
        const int dev_o = ggml_cuda_get_device();
        const int cc_o  = ggml_cuda_info().devices[dev_o].cc;
        if (GGML_CUDA_CC_IS_NVIDIA(cc_o) && cc_o < GGML_CUDA_CC_VOLTA && dev_o < GGML_CUDA_MAX_DEVICES &&
            (size_t) (dst->ne[0] / QK8_1) * sizeof(block_q8_1) <= QCACHE_CAP && qcache_ensure(g_qcache[dev_o], stream)) {
            qc_out = &g_qcache[dev_o];
            g_glu_q8_request = (block_q8_1 *) qcache_wbuf(*qc_out);
            g_glu_q8_done    = false;
        }
    }

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, q8_ptr, ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);

    if (qc_out) {
        if (g_glu_q8_done) {
            qcache_publish(*qc_out, dst, stream);
        }
        g_glu_q8_request = nullptr;
        g_glu_q8_done    = false;
    }
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}
