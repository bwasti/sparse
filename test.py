import numpy as np
import torch
import torch_sparse as ts

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

  def get_layout(self, i, ib_size, ob_size):
    # random sparsity map
    layout = np.zeros((ib_size, ob_size))
    while layout.sum() == 0:
      layout = np.ones((ib_size, ob_size))
      for _ in range(i):
          layout = np.random.randint(2, size=(ib_size,ob_size)) * layout
    print(f"Sparsity: {100*(1 - layout.sum() / layout.size):.2f}%")
    return layout

  def get_mask(self, layout, block_size):
    # TODO make this faster
    expanded_shape = layout.shape[0] * block_size, layout.shape[1] * block_size
    expanded_layout = np.zeros(expanded_shape, dtype=np.float)
    for i in range(expanded_shape[0]):
        for j in range(expanded_shape[1]):
            if layout[i // block_size, j // block_size]:
                expanded_layout[i, j] = 1
    mask = torch.tensor(expanded_layout, dtype=torch.float32, device=cuda)
    return mask

  def get_rand_vals(self, mb_size, i_size, o_size, block_size, return_mask=False):
    # blocked sizes
    ib_size=i_size//block_size
    ob_size=o_size//block_size
    layout = self.get_layout(2, ib_size, ob_size)
    X = torch.randn(mb_size, i_size, device=cuda)
    W = torch.randn(i_size, o_size, device=cuda)
    mask = self.get_mask(layout, block_size)
    masked_W = W * mask
    bs_W = ts.BlockSparseTensor(W, layout)
    if return_mask:
      return X, masked_W, bs_W, mask
    return X, masked_W, bs_W

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_fwd(self, mb, i, o, block_size):
    X, masked_W, bs_W = self.get_rand_vals(mb, i, o, block_size)
    Y = ts.mm(X, bs_W)
    Y_ref = X @ masked_W
    torch.testing.assert_allclose(Y, Y_ref)

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_dx(self, mb, i, o, block_size):
    X1, masked_W, bs_W = self.get_rand_vals(mb, i, o, block_size)
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
    X, masked_W, bs_W, mask = self.get_rand_vals(mb, i, o, block_size, True)

    masked_W.requires_grad = True
    bs_W.data.requires_grad = True

    ts.mm(X, bs_W).sum().backward()
    (X @ masked_W).sum().backward()
    torch.testing.assert_allclose(
      (masked_W.grad * mask).flatten().sum(),
      bs_W.data.grad.flatten().sum())

  @sweep(mb=[32,64,128], i=[32,64,128], o=[32,64,128], block_size=[4,8,16])
  def test_bwd_g_dx(self, mb, i, o, block_size):
    X1, _, bs_W, mask = self.get_rand_vals(mb, i, o, block_size, True)
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
    X, _, bs_W, mask = self.get_rand_vals(mb, i, o, block_size, True)

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

