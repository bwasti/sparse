import torch
import torch_sparse as ts
from test_util import get_rand_vals
import time


def bench_log(name, t, iters):
    d = time.time() - t
    t_iter = (d / iters) * 1e6
    print(f"{name}: {t_iter:.2f}us per iter")


def f(X, W):
    o = ts.mm(X, W)
    return o

def f_t(X, W):
    o = ts.mm(X, W, True)
    return o

def f_raw(X, W):
    o = ts.mm_raw(X, W)  # , True)
    return o

def b(X, Y, W):
    o = ts.mm_out(X, Y, W)
    return o


def bench_fwd(f, mb, i, o, block_size, sparsity, warmup=10, iters=100):
    print(f"{f.__name__}: {mb}x{i}x{o} bs {block_size}")
    X, masked_W, bs_W = get_rand_vals(mb, i, o, block_size, sparsity=sparsity)
    for _ in range(warmup):
        Y = torch.mm(X, masked_W)
    t = time.time()
    for _ in range(iters):
        Y = torch.mm(X, masked_W)
    bench_log("-- dense", t, iters)
    for _ in range(warmup):
        Y = f(X, bs_W)
    t = time.time()
    for _ in range(iters):
        Y = f(X, bs_W)
    bench_log("-- sparse", t, iters)


def bench_bwd(f, mb, i, o, block_size, sparsity, warmup=10, iters=100):
    print(f"{f.__name__}: {mb}x{i}x{o} bs {block_size}")
    X, masked_W, bs_W = get_rand_vals(mb, i, o, block_size, sparsity=sparsity)
    Y = torch.randn(mb, o, device=X.device, dtype=X.dtype)
    for _ in range(warmup):
        W = torch.mm(X.t(), Y)
    t = time.time()
    for _ in range(iters):
        W = torch.mm(X.t(), Y)
    bench_log("-- dense", t, iters)
    for _ in range(warmup):
        W = f(X, Y, bs_W)
    t = time.time()
    for _ in range(iters):
        W = f(X, Y, bs_W)
    bench_log("-- sparse", t, iters)


if __name__ == "__main__":
    for mb in [128, 512]:
        for i in [128, 512]:
            for o in [128, 512]:
                for bs in [8, 16, 32]:
                    for s in [3, 4]:
                        bench_fwd(f, mb, i, o, bs, s)
                        bench_fwd(f_t, mb, i, o, bs, s)
                        bench_fwd(f_raw, mb, i, o, bs, s)
                        bench_bwd(b, mb, i, o, bs, s)
