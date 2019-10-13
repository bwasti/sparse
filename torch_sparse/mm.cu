#include "mm.h"
#include <stdio.h>

__global__ void sparse_gemm_ehalf_kernel(
    const ehalf* __restrict__ A,
    const ehalf* __restrict__ B,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
          ehalf*              C,
    int64_t N, int64_t M, int64_t K, int64_t blocksize) {
}

__global__ void sparse_gemm_float_kernel(
    const float* __restrict__ X,
    const float* __restrict__ W,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
          float*              C,
    // N = minibatch, M = out channels
    int64_t N, int64_t M, int64_t K, int64_t blocksize) {
  extern __shared__ float C_block_accum[];
  // TODO prefetch LUT entries into shared
  uint4 lut_head = ((const uint4*)Lut)[blockIdx.x];
  uint nb = blockIdx.y; // minibatch block index
  if (nb * blocksize > N) return;
  uint lut_offset = lut_head.x;
  uint lut_size   = lut_head.y;
  uint idx_K      = lut_head.z; // output channel base index
  uint idx_Lock   = lut_head.w;
  uint offset_c = nb * M * blocksize + idx_K * blocksize;
  float* C_block = &C[offset_c];
  for (uint i = 0; i < blocksize*blocksize; ++i) {
    C_block_accum[i] = 0;
  }
  __syncthreads();

  if (lut_size > 0) {
    // accumulate segment along input channel space (cv.x)
    const uint2* segment = Lut + lut_offset;
    uint seg_i = 0;
    do {
      uint2 cv = segment[seg_i];
      uint offset_x = (nb * K + cv.x) * blocksize;
      uint offset_w = cv.y * blocksize * blocksize;
      const float* W_block = &W[offset_w];
      const float* X_block = &X[offset_x];
      for (uint i = 0; i < blocksize; ++i) {
        for (uint j = 0; j < blocksize; ++j) {
          float accum = 0;
          for (uint k = 0; k < blocksize; ++k) {
            accum += W_block[k * blocksize + i] * X_block[j * K + k];
          }
          C_block_accum[j * blocksize + i] += accum;
        }
      }
      seg_i++;
    } while (seg_i < lut_size);
  } else {
  }

  __syncthreads();
  for (uint i = 0; i < blocksize; ++i) {
    for (uint j = 0; j < blocksize; ++j) {
      C_block[i * M + j] = C_block_accum[i * blocksize + j];
    }
  }
}

void sparse_gemm_ehalf(
    const ehalf* X, const ehalf* W,
    const uint2* lut, 
    const float* gate,
    ehalf* C,
    int64_t N, int64_t M, int64_t K, int64_t blocksize
    ) {
    dim3 threads(16, 16);
    dim3 blocks(16, 16);
    // <<< blocks, threads, shared_mem, stream >>>
    sparse_gemm_ehalf_kernel<<<blocks, threads>>>(X, W, (const uint2*)lut, gate, C,
N, M, K, blocksize
);
}

void sparse_gemm_float(
    const float* A, const float* B,
    const uint2* lut, 
    const float* gate,
    float* C,
    int64_t N, int64_t M, int64_t K, 
    sparse_mm_params* params
    ) {
    dim3 threads(1);
    int64_t x = params->num_segs;
    int64_t y = N / params->block_size;
    dim3 blocks(x, y);
    size_t shared = params->block_size * params->block_size * sizeof(float);
    sparse_gemm_float_kernel<<<blocks, threads, shared>>>(A, B, (const uint2*)lut, gate, C, N, M, K, params->block_size);
}

__global__ void sparse_gemm_float_kernel_t(
    const float* __restrict__ dY,
    const float* __restrict__ W,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
          float*              C,
    // N = minibatch, M = out channels
    int64_t N, int64_t M, int64_t K, int64_t blocksize) {
  extern __shared__ float C_block_accum[];
  // TODO prefetch LUT entries into shared
  uint4 lut_head = ((const uint4*)Lut)[blockIdx.x];
  uint nb = blockIdx.y; // minibatch block index
  if (nb * blocksize > N) return;
  uint lut_offset = lut_head.x;
  uint lut_size   = lut_head.y;
  uint idx_K      = lut_head.z; // output channel base index
  uint idx_Lock   = lut_head.w;
  uint offset_c = (nb * M + idx_K) * blocksize;
  float* C_block = &C[offset_c];
  for (uint i = 0; i < blocksize*blocksize; ++i) {
    C_block_accum[i] = 0;
  }
  __syncthreads();

  if (lut_size > 0) {
    // accumulate segment along input channel space (cv.x)
    const uint2* segment = Lut + lut_offset;
    uint seg_i = 0;
    do {
      uint2 cv = segment[seg_i];
      uint offset_y = (nb * K + cv.x) * blocksize;
      uint offset_w = cv.y * blocksize * blocksize;
      const float* W_block = &W[offset_w];
      const float* dY_block = &dY[offset_y];
      for (uint i = 0; i < blocksize; ++i) {
        for (uint j = 0; j < blocksize; ++j) {
          float accum = 0;
          for (uint k = 0; k < blocksize; ++k) {
            accum += W_block[i * blocksize + k] * dY_block[j * K + k];
          }
          C_block_accum[j * blocksize + i] += accum;
        }
      }
      seg_i++;
    } while (seg_i < lut_size);
  } else {
  }

  __syncthreads();
  for (uint i = 0; i < blocksize; ++i) {
    for (uint j = 0; j < blocksize; ++j) {
      C_block[i * M + j] = C_block_accum[i * blocksize + j];
    }
  }
}

void sparse_gemm_float_t(
    const float* A, const float* B_t,
    const uint2* lut_t, 
    const float* gate,
    float* C,
    int64_t N, int64_t M, int64_t K, 
    sparse_mm_params* params
    ) {
    dim3 threads(1);
    int64_t x = params->num_segs;
    int64_t y = N / params->block_size;
    dim3 blocks(x, y);
    size_t shared = params->block_size * params->block_size * sizeof(float);
    sparse_gemm_float_kernel_t<<<blocks, threads, shared>>>(A, B_t, (const uint2*)lut_t, gate, C, N, M, K, params->block_size);
}

__global__ void sparse_gemm_float_kernel_tt(
    const float* __restrict__ X,
    const float* __restrict__ dY,
    const uint2* __restrict__ Lut,
    const float* __restrict__ Gate,
          float*              dW,
    // N = minibatch, M = out channels
    int64_t N, int64_t M, int64_t K, int64_t blocksize) {
  extern __shared__ float dW_block_accum[];
  // TODO prefetch LUT entries into shared
  uint2 lut_head = Lut[blockIdx.x];
  uint c = lut_head.x;
  uint k = lut_head.y;
  uint offset_dw = blockIdx.x * blocksize * blocksize;
  float* dW_block = &dW[offset_dw];
  for (uint i = 0; i < blocksize*blocksize; ++i) {
    dW_block_accum[i] = 0;
  }
  __syncthreads();

  // todo, ceildiv stuff
  for (uint n = 0; n < N / blocksize; ++n) {
    const float* X_block = &X[(n * K + c) * blocksize];
    const float* dY_block = &dY[(n * M + k) * blocksize];
    for (uint i = 0; i < blocksize; ++i) {
      for (uint j = 0; j < blocksize; ++j) {
        float accum = 0;
        for (uint k = 0; k < blocksize; ++k) {
          accum += X_block[k * K + j] * dY_block[k * M + i];
        }
        dW_block_accum[j * blocksize + i] += accum;
      }
    }
  }
  __syncthreads();
  // todo flatten
  for (uint i = 0; i < blocksize; ++i) {
    for (uint j = 0; j < blocksize; ++j) {
      dW_block[i * blocksize + j] = dW_block_accum[i * blocksize + j];
    }
  }
}

void sparse_gemm_float_tt(
    const float* X, const float* dY,
    const uint2* updat_lut, 
    const float* gate,
    float* dW,
    int64_t N, int64_t M, int64_t K, sparse_mm_params* params
    )  {
    dim3 threads(1);
    int64_t x = params->num_blocks;
    dim3 blocks(x);
    size_t shared = params->block_size * params->block_size * sizeof(float);
    sparse_gemm_float_kernel_tt<<<blocks, threads, shared>>>(X, dY, (const uint2*)updat_lut, gate, dW, N, M, K, params->block_size);
}
