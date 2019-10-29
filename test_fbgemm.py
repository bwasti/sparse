import torch
import torch.nn.functional as F
import torch_sparse
import scipy
import numpy as np
import time

cache_id = 0

def bench(p, n, m, k):
  global cache_id
  cache_id += 1
  #n = m = k = s
  a = torch.randn(n, k)
  sparsity = 1 / 1.5 ** p
  b = torch.tensor(scipy.sparse.random(k, m, density= sparsity).todense().astype(np.float32))
  print("{:.2f}% non-zero, n:{}xk:{}xm:{}".format(sparsity * 100, n, k, m))

  iters = 1000

  c_fbg = torch.ops.sparse2.smm(a, b, cache_id)
  torch.testing.assert_allclose(c_fbg, a@b)
  c_t = F.linear(a, b.t())
  torch.testing.assert_allclose(c_fbg, c_t)
  t = time.time()
  for _ in range(iters):
    c = torch.ops.sparse2.smm(a, b, cache_id)
  print(" ", time.time() - t)
  b_t = b.t().contiguous()
  t = time.time()
  for _ in range(iters):
    c = F.linear(a, b_t)
  print(" ", time.time() - t)

for p in range(1,10):
  for n in [1]:
    for m in [64, 128, 256]:
      for k in [128, 256]:
        bench(p, n, m, k)
