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
  flops = 2 * n * m * k
  iters = 1000

  c_fbg = torch.ops.sparse2.smm(a, b, cache_id)
  torch.testing.assert_allclose(c_fbg, a@b)
  c_t = F.linear(a, b.t())
  torch.testing.assert_allclose(c_fbg, c_t)
  t1_start = time.perf_counter()  
  for _ in range(iters):
    c = torch.ops.sparse2.smm(a, b, cache_id)
  d = time.perf_counter() - t1_start
  s = flops / (d * 1e9) * iters
  r = s * sparsity
  print(" ", d, s, r)

  b_t = b.t().contiguous()
  t1_start = time.perf_counter()  
  for _ in range(iters):
    #c = a @ b
    c = F.linear(a, b_t)
  d = time.perf_counter() - t1_start
  s = flops / (d * 1e9) * iters
  print(" ", d, s)
  #print(" ", time.perf_counter() - t1_start)

#n=m=k=1024
#bench(5, n, m, k)
#exit(0)
for p in range(5, 10):
  for n in [16, 64]:
    for m in [64, 128]:
      for k in [128, 256]:
        bench(p, n, m, k)
