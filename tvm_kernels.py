import tvm
import numpy as np
import os
import jinja2
import subprocess

class PTOp:
  def __init__(self, name, target, inputs, outputs, output_shapes, so_file, namespace="sparse"):
    self.namespace = namespace
    self.inputs = inputs
    self.outputs = list(zip(outputs, output_shapes))
    self.so = so_file
    self.cuda = target == "cuda"
    self.name = name

def compat_dtype(dtype):
  if dtype == "int32":
    return "int64_t"
  if dtype == "int64":
    return "int64_t"
  if dtype == "float32":
    return "double"
  if dtype == "float64":
    return "double"
def exact_dtype(dtype):
  if dtype == "int32":
    return "int32_t"
  if dtype == "int64":
    return "int64_t"
  if dtype == "float32":
    return "float"
  if dtype == "float64":
    return "double"

def build_for_pt(s, variables, tgt, target_host, name):
  f = tvm.build(s, variables, tgt, target_host=target_host, name=name)
  inputs = []
  input_shapes = {}
  outputs = []
  for var in variables:
    if hasattr(var, "op") and var.op in s.outputs:
      outputs.append(var)
    else:
      if hasattr(var, "shape"):
        for i, v in enumerate(var.shape):
          if v not in input_shapes:
            input_shapes[v] = (var.name, i)
      else:
        var.input_dtype = compat_dtype(var.dtype)
        var.real_dtype = exact_dtype(var.dtype)
        input_shapes[var.name] = (var.name, -1)
      inputs.append(var)
  print([(type(i), i.dtype) for i in inputs])
  output_shapes = []
  for output in outputs:
    o_shape = []
    for i, v in enumerate(output.shape):
      if v not in input_shapes:
        raise Exception(f"Can't handle dim {v} in {output}")
      o_shape.append(input_shapes[v])
    output_shapes.append(o_shape)
  path = "tvm_gen/"
  try:
    os.mkdir(path)
  except:
    pass
  so_file = os.path.join(path, f"{name}_{tgt}.so")
  f.export_library(so_file)
  with open("tvm_template.cpp", "r") as template_f:
    template = jinja2.Template(template_f.read())
    source = template.render(op=PTOp(name, tgt, inputs, outputs, output_shapes, so_file))

  # clang format stuff
  source = source.split('\n')
  source = filter(lambda x: x.strip()!="", source)
  source = "\n".join(source)
  cpp_file = os.path.join(path, f"{name}_{tgt}.cpp")
  with open(cpp_file, "w") as f:
    f.write(source)
  retcode=subprocess.call(["clang-format", "-i", "-style=Google", cpp_file])

def add():
  tgt_host="llvm"
  tgt="llvm"
  n = tvm.var("n")
  A = tvm.placeholder((n,), name='A', dtype="float32")
  B = tvm.placeholder((n,), name='B', dtype="float32")
  C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
  s = tvm.create_schedule(C.op)
  build_for_pt(s, [A, B, C], tgt, tgt_host, "myadd")

def mm():
  tgt_host="llvm"
  tgt="llvm"
  n = tvm.var("n")
  m = tvm.var("m")
  k = tvm.var("k")
  k_reduce = tvm.reduce_axis((0, k), "k_reduce")
  A = tvm.placeholder((n,k), name='A', dtype="float32")
  B = tvm.placeholder((k,m), name='B', dtype="float32")
  C = tvm.compute((n, m), lambda i, j: tvm.sum(A[i, k_reduce] * B[k_reduce, j], axis=k_reduce), name="C")
  s = tvm.create_schedule(C.op)
  build_for_pt(s, [A, B, C], tgt, tgt_host, "mymm")

add()
mm()

def nontensor():
  tgt_host="llvm"
  tgt="llvm"
  n = tvm.var("n")
  m = tvm.var("m", dtype="float32")
  A = tvm.placeholder((n,n), name='A', dtype="float32")
  B = tvm.compute(A.shape, lambda i, j: A[i,j] * m, name="B")
  s = tvm.create_schedule(B.op)
  build_for_pt(s, [A, m, B], tgt, tgt_host, "myscale")

nontensor()

