#include "mm.h"
#include "ATen/ATen.h"
#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/autograd/custom_function.h"
#include "torch/csrc/autograd/record_function.h"
#include <ATen/core/op_registration/op_registration.h>
#include <torch/script.h>

using namespace at;

namespace sparse {

at::Tensor bsmm(at::Tensor X, at::Tensor W, int64_t out, at::Tensor lut,
                int64_t block_size, int64_t num_segs, int64_t max_seg_len) {
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
    C = at::empty({N, M}, X.options().device(kCUDA).dtype(kFloat));
    sparse_mm_params params = {0};
    params.block_size = block_size;
    params.num_segs = num_segs;
    params.max_seg_len = max_seg_len;
    sparse_gemm_float(X.data_ptr<float>(), W.data_ptr<float>(),
                      (uint2 *)lut.data_ptr<int32_t>(), (float *)nullptr,
                      C.data_ptr<float>(), N, M, K, &params);
  }
  return C;
}

at::Tensor bsmm_t(at::Tensor dY, at::Tensor W, int64_t out, at::Tensor lut,
                  int64_t block_size, int64_t num_segs, int64_t max_seg_len) {
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
    C = at::empty({N, M}, dY.options().device(kCUDA).dtype(kFloat));
    sparse_mm_params params = {0};
    params.block_size = block_size;
    params.num_segs = num_segs;
    params.max_seg_len = max_seg_len;
    sparse_gemm_float_t(dY.data_ptr<float>(), W.data_ptr<float>(),
                        (uint2 *)lut.data_ptr<int32_t>(), (float *)nullptr,
                        C.data_ptr<float>(), N, M, K, &params);
  }
  return C;
}

at::Tensor mm_to_sparse(at::Tensor X, at::Tensor dY, at::Tensor lut,
                        int64_t block_size) {
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
    gemm_float_to_sparse(X.data_ptr<float>(), dY.data_ptr<float>(),
                         (uint2 *)lut.data_ptr<int32_t>(), (float *)nullptr,
                         dW.data_ptr<float>(), N, M, K, &params);
  }
  return dW;
}
} // namespace sparse

/*

class BSMM(Function):
    @staticmethod
    def forward( ctx, X, W, flut, blut, ulut,
        o_size, block_size, num_fsegs, max_fseg, num_bsegs, max_bseg, t,
    ):
        ctx.save_for_backward(X, W, flut, blut, ulut)
        ctx.block_size = block_size
        ctx.num_fsegs  = num_fsegs
        ctx.max_fseg   = max_fseg
        ctx.num_bsegs  = num_bsegs
        ctx.max_bseg   = max_bseg
        if t:
            o = torch.ops.sparse.bsmm_t(
                X, W, o_size, flut, block_size, num_fsegs, max_fseg
            )
        else:
            o = torch.ops.sparse.bsmm(
                X, W, o_size, flut, block_size, num_fsegs, max_fseg
            )
        return o

    @staticmethod
    def backward(ctx, dY):
        X, W, flut, blut, ulut = ctx.saved_tensors
        dX = None
        if ctx.needs_input_grad[0]:
          dX = BSMM.apply( dY, W, blut, flut, ulut,
              X.shape[1], ctx.block_size, ctx.num_bsegs, ctx.max_bseg,
ctx.num_fsegs, ctx.max_fseg, True,
          )
        dW = None
        if ctx.needs_input_grad[1]:
          dW = BSMM_out.apply( X, dY, ulut, flut, blut,
              ctx.block_size, ctx.num_fsegs, ctx.max_fseg, ctx.num_bsegs,
ctx.max_bseg,) return dX, dW, None, None, None, None, None, None, None, None,
None, None

class BSMM_out(Function):
    @staticmethod
    def forward( ctx, X, dY, ulut, flut, blut,
        block_size, num_fsegs, max_fseg, num_bsegs, max_bseg,
    ):
        ctx.save_for_backward( X, dY, flut, blut, ulut,)
        ctx.block_size   = block_size
        ctx.num_fsegs    = num_fsegs
        ctx.max_fseg     = max_fseg
        ctx.num_bsegs    = num_bsegs
        ctx.max_bseg     = max_bseg
        dW = torch.ops.sparse.mm_to_bs(X, dY, ulut, block_size)
        return dW

    @staticmethod
    def backward(ctx, ddW):
        ( X, dY, flut, blut, ulut,) = ctx.saved_tensors
        dXT = BSMM.apply( dY, ddW, blut, flut, ulut,
            X.shape[1], ctx.block_size, ctx.num_bsegs, ctx.max_bseg,
ctx.num_fsegs, ctx.max_fseg, True,) ddY = BSMM.apply( X, ddW, flut, blut, ulut,
dY.shape[1], ctx.block_size, ctx.num_fsegs, ctx.max_fseg, ctx.num_bsegs,
ctx.max_bseg, True,) return dXT, ddY, None, None, None, None, None, None, None,
None, None
*/

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class BlockSparseMatMulOut
    : public torch::autograd::Function<BlockSparseMatMulOut> {
public:
  static variable_list forward(AutogradContext *ctx, Variable X, Variable dY,
                               Variable ulut, Variable flut, Variable blut,
                               const int64_t block_size,
                               const int64_t num_fsegs, const int64_t max_fseg,
                               const int64_t num_bsegs, const int64_t max_bseg);
  static variable_list backward(AutogradContext *ctx, variable_list ddW);
};

class BlockSparseMatMul : public torch::autograd::Function<BlockSparseMatMul> {
public:
  static variable_list forward(AutogradContext *ctx, Variable X, Variable W,
                               Variable flut, Variable blut, Variable ulut,
                               const int64_t out_size, const int64_t block_size,
                               const int64_t num_fsegs, const int64_t max_fseg,
                               const int64_t num_bsegs, const int64_t max_bseg,
                               const bool transpose) {
    ctx->saved_data["out_size"] = out_size;
    ctx->saved_data["block_size"] = block_size;
    ctx->saved_data["num_fsegs"] = num_fsegs;
    ctx->saved_data["max_fseg"] = max_fseg;
    ctx->saved_data["num_bsegs"] = num_bsegs;
    ctx->saved_data["max_bseg"] = max_bseg;
    ctx->saved_data["transpose"] = transpose;
    ctx->save_for_backward({X, W, flut, blut, ulut});
    if (transpose) {
      return {sparse::bsmm_t(X, W, out_size, flut, block_size, num_fsegs,
                             max_fseg)};
    }
    return {
        sparse::bsmm(X, W, out_size, flut, block_size, num_fsegs, max_fseg)};
  }

  static variable_list backward(AutogradContext *ctx, variable_list dY) {
    auto saved = ctx->get_saved_variables();
    Variable dX = BlockSparseMatMul::apply(
        dY[0], saved[1] /*W*/, saved[3] /*blut*/, saved[2] /*flut*/,
        saved[4] /*ulut*/, saved[0].sizes()[1],
        ctx->saved_data["block_size"].toInt(),
        ctx->saved_data["num_bsegs"].toInt(),
        ctx->saved_data["max_bseg"].toInt(),
        ctx->saved_data["num_fsegs"].toInt(),
        ctx->saved_data["max_fseg"].toInt(),
        !ctx->saved_data["transpose"].toBool())[0];
    Variable dW = BlockSparseMatMulOut::apply(
        saved[0] /*X*/, dY[0], saved[4] /*ulut*/, saved[2] /*flut*/,
        saved[3] /*blut*/, ctx->saved_data["block_size"].toInt(),
        ctx->saved_data["num_fsegs"].toInt(),
        ctx->saved_data["max_fseg"].toInt(),
        ctx->saved_data["num_bsegs"].toInt(),
        ctx->saved_data["max_bseg"].toInt())[0];
    return {dX,         dW,         Variable(), Variable(),
            Variable(), Variable(), Variable(), Variable(),
            Variable(), Variable(), Variable(), Variable()};
  }
};

variable_list BlockSparseMatMulOut::forward(
    AutogradContext *ctx, Variable X, Variable dY, Variable ulut, Variable flut,
    Variable blut, const int64_t block_size, const int64_t num_fsegs,
    const int64_t max_fseg, const int64_t num_bsegs, const int64_t max_bseg) {
  ctx->saved_data["block_size"] = block_size;
  ctx->saved_data["num_fsegs"] = num_fsegs;
  ctx->saved_data["max_fseg"] = max_fseg;
  ctx->saved_data["num_bsegs"] = num_bsegs;
  ctx->saved_data["max_bseg"] = max_bseg;
  ctx->save_for_backward({X, dY, flut, blut, ulut});
  return {sparse::mm_to_sparse(X, dY, ulut, block_size)};
}

variable_list BlockSparseMatMulOut::backward(AutogradContext *ctx,
                                             variable_list ddW) {
  auto saved = ctx->get_saved_variables();
  Variable dXT = BlockSparseMatMul::apply(
      saved[1] /*dY*/, ddW[0], saved[3] /*blut*/, saved[2] /*flut*/,
      saved[4] /*ulut*/, saved[0].sizes()[1],
      ctx->saved_data["block_size"].toInt(),
      ctx->saved_data["num_bsegs"].toInt(), ctx->saved_data["max_bseg"].toInt(),
      ctx->saved_data["num_fsegs"].toInt(), ctx->saved_data["max_fseg"].toInt(),
      true)[0];
  Variable ddY = BlockSparseMatMul::apply(
      saved[0] /*X*/, ddW[0], saved[2] /*flut*/, saved[3] /*blut*/,
      saved[4] /*ulut*/, saved[1].sizes()[1],
      ctx->saved_data["block_size"].toInt(),
      ctx->saved_data["num_fsegs"].toInt(), ctx->saved_data["max_fseg"].toInt(),
      ctx->saved_data["num_bsegs"].toInt(), ctx->saved_data["max_bseg"].toInt(),
      true)[0];
  return {dXT,        ddY,        Variable(), Variable(),
          Variable(), Variable(), Variable(), Variable(),
          Variable(), Variable(), Variable()};
}

at::Tensor bsmm_autograd(at::Tensor X, at::Tensor W, at::Tensor flut,
                         at::Tensor blut, at::Tensor ulut,
                         const int64_t out_size, const int64_t block_size,
                         const int64_t num_fsegs, const int64_t max_fseg,
                         const int64_t num_bsegs, const int64_t max_bseg,
                         bool transpose) {
  return BlockSparseMatMul::apply(X, W, flut, blut, ulut, out_size, block_size,
                                  num_fsegs, max_fseg, num_bsegs, max_bseg,
                                  transpose)[0];
}

at::Tensor mm_to_sparse_autograd(
    at::Tensor X, at::Tensor dY, at::Tensor ulut, at::Tensor flut,
    at::Tensor blut, const int64_t block_size, const int64_t num_fsegs,
    const int64_t max_fseg, const int64_t num_bsegs, const int64_t max_bseg) {
  return BlockSparseMatMulOut::apply(X, dY, ulut, flut, blut, block_size,
                                     num_fsegs, max_fseg, num_bsegs,
                                     max_bseg)[0];
}

static auto registry0 =
    torch::RegisterOperators()
        .op("sparse::bsmm", torch::RegisterOperators::options().kernel(
                                c10::TensorTypeId::CUDATensorId, bsmm_autograd))
        .op("sparse::bsmm_raw",
            torch::RegisterOperators::options().kernel(
                c10::TensorTypeId::CUDATensorId, sparse::bsmm))
        .op("sparse::mmbs",
            torch::RegisterOperators::options().kernel(
                c10::TensorTypeId::CUDATensorId, mm_to_sparse_autograd));
