#include "ATen/ATen.h"
#include <ATen/core/op_registration/op_registration.h>
#include "fbgemm/FbgemmSpMM.h"

#include <functional>

using namespace at;

namespace sparse2 {

using SpMMFloatFn = std::function< void( const float*, float*, std::uint64_t flags) >;

static std::unordered_map<int64_t, SpMMFloatFn > cache;

at::Tensor smm(at::Tensor X, at::Tensor W, int64_t cache_id) {
  auto M = X.sizes()[0];
  auto K = X.sizes()[1];
  auto N = W.sizes()[1];
  at::Tensor Y = at::empty({N, M}, X.options());

  if (cache.find(cache_id) == cache.end()) {
    cache[cache_id] = fbgemm::generateSpMM_avx512<float>(
        (int)N,
        (int)M,
        (int)K,
        (float*)W.t().contiguous().data_ptr(),
        (int)K,
        (int)M,
        (int)M);
  }
  auto& fn = cache.at(cache_id);
  fn((float*)X.t().contiguous().data_ptr(), (float*)Y.data_ptr(), 0 /* flag */);
  return Y.t();
}

}

static auto registry0 = torch::RegisterOperators()
  .op("sparse2::smm", torch::RegisterOperators::options().kernel(
        c10::TensorTypeId::CPUTensorId, sparse2::smm));
