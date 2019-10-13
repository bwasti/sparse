import torch
import torch_sparse as ts
import numpy as np

cuda = torch.device('cuda')

def get_layout(i, ib_size, ob_size):
  # random sparsity map
  layout = np.zeros((ib_size, ob_size))
  while layout.sum() == 0:
    layout = np.ones((ib_size, ob_size))
    for _ in range(i):
        layout = np.random.randint(2, size=(ib_size,ob_size)) * layout
  print(f"sparsity: {100*(1 - layout.sum() / layout.size):.2f}%")
  return layout

def get_mask(layout, block_size):
  # TODO make this faster
  expanded_shape = layout.shape[0] * block_size, layout.shape[1] * block_size
  expanded_layout = np.zeros(expanded_shape, dtype=np.float)
  for i in range(expanded_shape[0]):
      for j in range(expanded_shape[1]):
          if layout[i // block_size, j // block_size]:
              expanded_layout[i, j] = 1
  mask = torch.tensor(expanded_layout, dtype=torch.float32, device=cuda)
  return mask

def get_rand_vals( mb_size, i_size, o_size, block_size, return_mask=False, sparsity=2):
  # blocked sizes
  ib_size=i_size//block_size
  ob_size=o_size//block_size
  layout = get_layout(sparsity, ib_size, ob_size)
  X = torch.randn(mb_size, i_size, device=cuda)
  W = torch.randn(i_size, o_size, device=cuda)
  mask = get_mask(layout, block_size)
  masked_W = W * mask
  bs_W = ts.BlockSparseTensor(W, layout)
  if return_mask:
    return X, masked_W, bs_W, mask
  return X, masked_W, bs_W
