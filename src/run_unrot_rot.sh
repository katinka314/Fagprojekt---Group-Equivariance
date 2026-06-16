#!/bin/sh
### LSF job script: train-on-unrotated / test-on-rotated, CNN vs GE-CNN, on a DTU GPU.
### Submit from the project root with:   bsub < src/run_unrot_rot.sh
### Queue: gpuv100 has the most free slots; swap to gpua100/gpul40s/gpuh100 if preferred.
#BSUB -q gpuv100
#BSUB -J unrot_rot
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o reports/unrot_rot/lsf_%J.out
#BSUB -e reports/unrot_rot/lsf_%J.err

cd /zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance

export N_EPOCHS=${N_EPOCHS:-100}
# DEVICE unset -> the script auto-detects CUDA on the GPU node.

nvidia-smi || true
~/.local/bin/uv run python src/train_unrot_test_rot.py
