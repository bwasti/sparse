#include "ATen/ATen.h"

struct COOTensor {
  at::Tensor vals; // float or whatever
  at::Tensor lut; // 64bit int
  at::Tensor lut_dx; // 64bit int
  at::Tensor lut_dw; // 64bit int
};

