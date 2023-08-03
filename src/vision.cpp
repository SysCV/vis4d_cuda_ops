/*!
**************************************************************************************************
* Vis4D CUDA Operations
**************************************************************************************************
* Modified from https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/src/vision.cpp
**************************************************************************************************
*/
#include "nms_rotated.h"
#include "ms_deform_attn.h"
#include "deform_conv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_rotated", &nms_rotated);
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &modulated_deform_conv_backward,
      "modulated_deform_conv_backward");
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
