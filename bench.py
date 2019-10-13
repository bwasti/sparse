import torch
import torch_sparse as ts
from test_util import get_rand_vals
import time


def bench_log(name, t, iters):
    d = time.time() - t
    t_iter = (d / iters) * 1e6
    print(f"{name}: {t_iter:.2f}us per iter")


# For some reason, torch.autograd.Function.apply is really slow ¯\_(ツ)_/¯
def f(X, W):
    o = torch.ops.sparse.bsmm(
        X, W.data, W.shape[1], W.flut, W.block_size, W.num_fsegs, W.max_fseg
    )
    return o


def bench_fwd(mb, i, o, block_size, sparsity, warmup=10, iters=100):
    print(f"{mb}x{i}x{o} bs {block_size}")
    X, masked_W, bs_W = get_rand_vals(mb, i, o, block_size, sparsity=sparsity)
    for _ in range(warmup):
        Y = torch.mm(X, masked_W)
    t = time.time()
    for _ in range(iters):
        Y = torch.mm(X, masked_W)
    bench_log("dense", t, iters)
    for _ in range(warmup):
        # Y = ts.mm(X, bs_W)
        Y = f(X, bs_W)
    t = time.time()
    for _ in range(iters):
        # Y = ts.mm(X, bs_W)
        Y = f(X, bs_W)
    bench_log("sparse", t, iters)


if __name__ == "__main__":
    for mb in [128, 512]:
        for i in [128, 512]:
            for o in [128, 512]:
                for bs in [8, 16, 32]:
                    for s in [3, 4]:
                        bench_fwd(mb, i, o, bs, s)
