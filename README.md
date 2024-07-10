# Vis4D Cuda Operations

## Installation

You can directly install with pip and set `TORCH_CUDA_ARCH_LIST` to specify the cuda architecture if needed.
```bash
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

pip install -v .
```

## Usage
```python
import torch
from vis4d_cuda_ops import ms_deform_attn_forward, ms_deform_attn_backward
...
```

## Add a new Op:
1. Add cuda and cpu ops.
2. Delcare its Python interface in `src/vision.cpp`.
