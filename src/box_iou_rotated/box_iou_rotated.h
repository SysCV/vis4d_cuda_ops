// Copyright (c) Facebook, Inc. and its affiliates.
// modified from
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
#pragma once
#include <torch/types.h>

at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return box_iou_rotated_cuda(boxes1.contiguous(), boxes2.contiguous());
#else
    AT_ERROR("Not compiled with GPU support!");
#endif
  }

  return box_iou_rotated_cpu(boxes1.contiguous(), boxes2.contiguous());
}
