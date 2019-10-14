#include "mm.h"
#include <ATen/core/op_registration/op_registration.h>
#include "ATen/ATen.h"
#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/autograd/record_function.h"

using namespace at;

static auto registry0 =
    torch::RegisterOperators()
        .op("sparse::bsmm",
            torch::RegisterOperators::options().kernel(
                c10::TensorTypeId::CUDATensorId,
                [](at::Tensor X, at::Tensor W, int64_t out, at::Tensor lut,
                   int64_t block_size, int64_t num_segs,
                   int64_t max_seg_len) -> at::Tensor {
                  at::cuda::CUDAGuard device_guard(X.device());
                  std::vector<c10::IValue> inp{X, W, lut};
                  RECORD_FUNCTION("sparse::bsmm", inp);

                  TORCH_CHECK(X.dim() == 2);
                  if (!X.is_contiguous()) {
                    // std::cerr << "non-contiguous X, likely slowdown\n";
                    X = X.contiguous();
                  }
                  auto N = X.sizes()[0];
                  auto K = X.sizes()[1];
                  auto M = out;
                  at::Tensor C;
                  TORCH_CHECK(X.dtype() == kFloat);
                  TORCH_CHECK(lut.dtype() == kInt);
                  TORCH_CHECK(X.device() == W.device());
                  TORCH_CHECK(X.device() == lut.device());
                  if (X.dtype() == kHalf) {
                    // TODO
                  }
                  if (X.dtype() == kFloat) {
                    C = at::empty({N, M},
                                  X.options().device(kCUDA).dtype(kFloat));
                    sparse_mm_params params = {0};
                    params.block_size = block_size;
                    params.num_segs = num_segs;
                    params.max_seg_len = max_seg_len;
                    sparse_gemm_float(X.data_ptr<float>(), W.data_ptr<float>(),
                                      (uint2*)lut.data_ptr<int32_t>(),
                                      (float*)nullptr, C.data_ptr<float>(), N,
                                      M, K, &params);
                  }
                  return C;
                }))
        .op("sparse::bsmm_t",
            torch::RegisterOperators::options().kernel(
                c10::TensorTypeId::CUDATensorId,
                [](at::Tensor dY, at::Tensor W, int64_t out, at::Tensor lut,
                   int64_t block_size, int64_t num_segs,
                   int64_t max_seg_len) -> at::Tensor {
                  TORCH_CHECK(dY.dim() == 2);
                  at::cuda::CUDAGuard device_guard(dY.device());
                  std::vector<c10::IValue> inp{dY, W, lut};
                  RECORD_FUNCTION("sparse::bsmm_t", inp);
                  auto N = dY.sizes()[0];
                  auto K = dY.sizes()[1];
                  auto M = out;
                  at::Tensor C;
                  if (!dY.is_contiguous()) {
                    // std::cerr << "non-contiguous dY, likely slowdown\n";
                    dY = dY.contiguous();
                  }
                  if (!W.is_contiguous()) {
                    // std::cerr << "non-contiguous W, likely slowdown\n";
                    W = W.contiguous();
                  }
                  TORCH_CHECK(dY.dtype() == kFloat);
                  TORCH_CHECK(lut.dtype() == kInt);
                  TORCH_CHECK(dY.device() == W.device());
                  TORCH_CHECK(dY.device() == lut.device());
                  if (dY.dtype() == kHalf) {
                    // TODO
                  }
                  if (dY.dtype() == kFloat) {
                    C = at::empty({N, M},
                                  dY.options().device(kCUDA).dtype(kFloat));
                    sparse_mm_params params = {0};
                    params.block_size = block_size;
                    params.num_segs = num_segs;
                    params.max_seg_len = max_seg_len;
                    sparse_gemm_float_t(
                        dY.data_ptr<float>(), W.data_ptr<float>(),
                        (uint2*)lut.data_ptr<int32_t>(), (float*)nullptr,
                        C.data_ptr<float>(), N, M, K, &params);
                  }
                  return C;
                }))
        .op("sparse::mm_to_bs",
            torch::RegisterOperators::options().kernel(
                c10::TensorTypeId::CUDATensorId,
                [](at::Tensor X, at::Tensor dY, at::Tensor lut,
                   int64_t block_size) -> at::Tensor {
                  at::cuda::CUDAGuard device_guard(X.device());
                  std::vector<c10::IValue> inp{X, dY, lut};
                  RECORD_FUNCTION("sparse::mm_to_bs", inp);
                  if (!dY.is_contiguous()) {
                    // std::cerr << "non-contiguous dY, likely slowdown\n";
                    dY = dY.contiguous();
                  }
                  if (!X.is_contiguous()) {
                    // std::cerr << "non-contiguous X, likely slowdown\n";
                    X = X.contiguous();
                  }
                  TORCH_CHECK(dY.dim() == 2);
                  TORCH_CHECK(X.dim() == 2);
                  TORCH_CHECK(lut.dtype() == kInt);
                  auto N = X.sizes()[0];
                  TORCH_CHECK(dY.sizes()[0] == N);
                  auto M = dY.sizes()[1];
                  auto K = X.sizes()[1];
                  int64_t num_blocks = lut.numel() / 2;
                  at::Tensor dW;
                  TORCH_CHECK(dY.dtype() == kFloat);
                  TORCH_CHECK(dY.device() == X.device());
                  TORCH_CHECK(dY.device() == lut.device());
                  if (dY.dtype() == kHalf) {
                    // TODO
                  }
                  if (dY.dtype() == kFloat) {
                    dW = at::empty({num_blocks, block_size, block_size},
                                   dY.options().device(kCUDA).dtype(kFloat));
                    sparse_mm_params params = {0};
                    params.block_size = block_size;
                    params.num_segs = 0;
                    params.max_seg_len = 0;
                    params.num_blocks = num_blocks;
                    gemm_float_to_sparse(
                        X.data_ptr<float>(), dY.data_ptr<float>(),
                        (uint2*)lut.data_ptr<int32_t>(), (float*)nullptr,
                        dW.data_ptr<float>(), N, M, K, &params);
                  }
                  return dW;
                }));
