#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

/* TODO */
typedef struct __align__(2) ehalf {
  __device__ __forceinline__ ehalf() {}
  __device__ __forceinline__ ehalf(const unsigned short val) : x(val) {}
  unsigned short x;
}
ehalf;

struct sparse_mm_params {
  int64_t block_size;
  int64_t num_blocks;
  int64_t num_segs;
  int64_t max_seg_len;
};

void sparse_gemm_float(const float *A, const float *B, const uint2 *lut,
                       const float *gate, float *C, int64_t N, int64_t M,
                       int64_t K, sparse_mm_params *params);

void sparse_gemm_float_t(const float *A, const float *B_t, const uint2 *lut_t,
                         const float *gate, float *C, int64_t N, int64_t M,
                         int64_t K, sparse_mm_params *params);

void gemm_float_to_sparse(const float *X, const float *dY,
                          const uint2 *updat_lut, const float *gate, float *dW,
                          int64_t N, int64_t M, int64_t K,
                          sparse_mm_params *params);
