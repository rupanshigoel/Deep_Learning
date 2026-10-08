#!/bin/bash
echo "=== module avail cuda ==="
module avail 2>&1 | grep -iE "cuda|cudnn" | head -40
echo
echo "=== /home/apps dirs ==="
ls /home/apps | head -30
echo
echo "=== /home/apps/compiler/cuda ==="
ls /home/apps/compiler/cuda 2>&1
echo
echo "=== trying module load compiler/cuda/11.8/compilervars ==="
module load compiler/cuda/11.8/compilervars 2>&1 | head -20
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "PATH (cuda bits)=$(echo $PATH | tr ':' '\n' | grep -i cuda)"
echo
echo "=== import torch test ==="
$HOME/.conda/envs/col775_t25/bin/python -c "
import torch, torchvision
print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'built_with', torch.version.cuda)
print('torchvision', torchvision.__version__)
" 2>&1
