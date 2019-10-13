import numpy as np
import torch
import torch_sparse as ts
from test_util import get_rand_vals

import unittest
import random

sweep_size = 10
def sweep(**kwargs):
  def wrapped_f(f):
    swept_args = []
    for _ in range(sweep_size):
      args = {}
      for arg in kwargs:
        a = kwargs[arg]
        if type(a) is list:
          args[arg] = random.choice(a)
        else:
          args[arg] = a
      swept_args.append(args)
    def f_(*args, **kwargs):
      for kw in swept_args:
        print("running", f.__name__, "with", kw)
        f(*args, **kw)
    return f_
  return wrapped_f

cuda = torch.device('cuda')

class TestBlockSparseTensor(unittest.TestCase):

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_fwd(self, mb, i, o, block_size):
    X, masked_W, bs_W = get_rand_vals(mb, i, o, block_size)
    Y = ts.mm(X, bs_W)
    Y_ref = X @ masked_W
    torch.testing.assert_allclose(Y, Y_ref)

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_dx(self, mb, i, o, block_size):
    X1, masked_W, bs_W = get_rand_vals(mb, i, o, block_size)
    X2 = X1.clone()
    X1.requires_grad = True
    X2.requires_grad = True

    Y = ts.mm(X1, bs_W)
    Y.sum().backward()
    Y_ref = X2 @ masked_W
    Y_ref.sum().backward()
    torch.testing.assert_allclose(Y, Y_ref)
    torch.testing.assert_allclose(X1.grad, X2.grad)

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_dw(self, mb, i, o, block_size):
    X, masked_W, bs_W, mask = get_rand_vals(mb, i, o, block_size, True)

    masked_W.requires_grad = True
    bs_W.data.requires_grad = True

    ts.mm(X, bs_W).sum().backward()
    (X @ masked_W).sum().backward()
    torch.testing.assert_allclose(
      (masked_W.grad * mask).flatten().sum(),
      bs_W.data.grad.flatten().sum())

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_g_dx(self, mb, i, o, block_size):
    X1, _, bs_W, mask = get_rand_vals(mb, i, o, block_size, True)
    X2 = X1.clone()
    X1.requires_grad = True
    X2.requires_grad = True

    Y = torch.randn(mb, o, device=cuda)

    bs_W = ts.mm_out(X1.t(), Y, bs_W)
    W = (X2.t() @ Y) * mask
    torch.testing.assert_allclose(bs_W.data.sum(), W.sum())

    bs_W.data.sum().backward()
    W.sum().backward()
    torch.testing.assert_allclose(X1.grad, X2.grad)

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_g_dy(self, mb, i, o, block_size):
    X, _, bs_W, mask = get_rand_vals(mb, i, o, block_size, True)

    Y1 = torch.randn(mb, o, device=cuda)
    Y2 = Y1.clone()
    Y1.requires_grad = True
    Y2.requires_grad = True

    bs_W = ts.mm_out(X.t(), Y1, bs_W)
    W = (X.t() @ Y2) * mask
    torch.testing.assert_allclose(bs_W.data.sum(), W.sum())

    bs_W.data.sum().backward()
    W.sum().backward()
    torch.testing.assert_allclose(Y1.grad, Y2.grad)


if __name__ == "__main__":
  unittest.main()

