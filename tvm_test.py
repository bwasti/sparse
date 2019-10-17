import torch
import torch_sparse

a = torch.randn(10)
b = torch.randn(10)
c = torch.ops.sparse.myadd(a, b)
print(c)
print(a + b)

a = torch.randn(10,10)
b = torch.randn(10,10)
c = torch.ops.sparse.mymm(a,b)
print(c)
print(a @ b)

a = torch.randn(10,10)
scale = 3.4
c = torch.ops.sparse.myscale(a, scale)
print(c)
print(a * scale)
