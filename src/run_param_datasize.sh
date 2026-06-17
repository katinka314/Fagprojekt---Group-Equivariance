#!/bin/sh
### LSF job: parameter- and data-efficiency sweeps, GE-CNN vs CNN, multi-seed.
### Remember: set SMOKE = False in test_parameters_datasize_v2.py before submitting.
### Submit from the project root with:   bsub < src/run_param_datasize.sh
#BSUB -q gpuv100
#BSUB -J param_datasize
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 18:00
#BSUB -o reports/param_datasize/lsf_%J.out
#BSUB -e reports/param_datasize/lsf_%J.err

cd /zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
mkdir -p reports/param_datasize

nvidia-smi || true
~/.local/bin/uv run python src/test_parameters_datasize_v2.py
