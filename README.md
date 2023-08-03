# Vis4D Cuda Operations

## Installation
### Requirements
```bash
pip install -r requirements.txt
```

### Build
```bash
bash make.sh
```
If you use python `venv `, you can add `--prefix` to specify the installation path.
```bash
bash make.sh --prefix $VIRTUAL_ENV
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
