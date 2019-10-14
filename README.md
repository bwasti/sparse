# sparse

(Experimental) Sparse tensor addons. Currently supports fp32 on CUDA.

# Build + Test

```
pip install torch scipy
./build.sh
PYTHONPATH=build python test.py
```

# Usage

There are three API entry points

```python
# weight[fp32], layout[int32] -> blocksparse
bs_W = ts.BlockSparseTensor(W, layout)

# input[fp32], blocksparse -> output[fp32]
Y = ts.mm(X, bs_W)

# in_channels, out_channels, sparsity, block_size -> nn.Module
fc = ts.SparseLinear(128, 512, 0.90, 16)
```

Full example

```python
import torch
import numpy as np
import torch_sparse as ts

minibatch = 64
input_features = 128
output_features = 512
block_size = 16

# GPU only
cuda = torch.device('cuda')
X = torch.randn(minibatch, input_features, device=cuda)
W = torch.randn(input_features, output_features, device=cuda)

# First, generate a sparse layout
ib = input_features // block_size
ob = output_features // block_size
layout = np.random.randint(2, size=(ib,ob))

# Then, create a blocksparse weight
bs_W = ts.BlockSparseTensor(W, layout)

# Differentiable matrix multiplication
Y = ts.mm(X, bs_W)
```

# TODO

- CPU kernels
- Improve GPU perf
- Add fp16 support
- Full model test + training scripts
