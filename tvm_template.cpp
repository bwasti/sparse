#include "c10/cuda/CUDAGuard.h"
#include "torch/csrc/autograd/custom_function.h"
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/DLConvertor.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

// TODO: non tensor inputs
// TODO: multiple outputs

{% if op.outputs|length == 1 %}
at::Tensor 
{% else %}
#error Can't support multiple out yet
{% endif %}
{{ op.name }}_impl(
{% for input in op.inputs %}
  {% if input.shape %}
  at::Tensor {{ input.name }}
  {% else %}
  {{ input.input_dtype }} {{ input.name }}
  {% endif %}
  {% if not loop.last %}
  ,
  {% endif %}
{% endfor %}
) {
  {% if op.cuda %}
  at::cuda::CUDAGuard device_guard(X.device());
  {% endif %}
{% for input in op.inputs %}
  {% if input.shape %}
  auto dl_{{ input.name }} = at::toDLPack({{ input.name }});
  {% else %}
  auto dl_{{ input.name }} = static_cast<{{input.real_dtype}}>({{input.name}});
  {% endif %}
{% endfor %}
{% for output, shape in op.outputs %}
  {% for dim in shape %}
  auto {{output.name}}_shape_{{loop.index}} = {{dim[0]}}.sizes()[{{dim[1]}}];
  {% endfor %}
  auto {{output.name}} = at::empty(
  {
  {% for dim in shape %}
    {{output.name}}_shape_{{loop.index}}
    {% if not loop.last %}
    ,
    {% endif %}
  {% endfor %}
  },
  {{op.inputs[0].name}}.options()
  );
  auto dl_{{output.name}} = at::toDLPack({{output.name}});
{% endfor %}
  tvm::runtime::Module mod =
        tvm::runtime::Module::LoadFromFile("{{op.so}}");
  tvm::runtime::PackedFunc f = mod.GetFunction("{{op.name}}");
  f(
{% for input in op.inputs %}
  dl_{{ input.name }},
{% endfor %}
{% for output, _ in op.outputs %}
  dl_{{output.name}}
  {% if not loop.last %}
  ,
  {% endif %}
{% endfor %}
  );
  return {{op.outputs[0][0].name}};
}

static auto registry = torch::RegisterOperators()
  .op("{{ op.namespace }}::{{ op.name }}",
    torch::RegisterOperators::options().kernel(
    {% if op.cuda %}
          c10::TensorTypeId::CUDATensorId
    {% else %}
          c10::TensorTypeId::CPUTensorId
    {% endif %}
    , {{ op.name }}_impl));

