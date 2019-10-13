import torch
import numpy as np
import torch_sparse as ts

minibatch = 64
input_features = 128
output_features = 512
block_size = 16

cuda = torch.device("cuda")
X = torch.randn(minibatch, input_features, device=cuda)
W = torch.randn(input_features, output_features, device=cuda)

# First, generate a sparse layout
ib = input_features // block_size
ob = output_features // block_size
layout = np.random.randint(2, size=(ib, ob))

# Then, create a blocksparse weight
bs_W = ts.BlockSparseTensor(W, layout)

# Differentiable matrix multiplication
Y = ts.mm(X, bs_W)
