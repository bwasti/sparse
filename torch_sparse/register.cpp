// Everything is statically linked in -- this file is just for pybind

#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_torch_sparse, m) {}
