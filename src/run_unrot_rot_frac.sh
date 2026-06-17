#!/bin/sh
### Train-unrot / test-rot DATASIZE SWEEP: one job per TRAIN_FRACTION (set via env at submit).
### Outputs land in reports/unrot_rot/frac_<f>/ (PNGs, aggregate.json/pt, LSF logs).
### GE-CNN is param-matched to CNN (l=3, ch=8, rings=4 ~ 133k ~ CNN ch=16 128k).
### Submit ALL fractions with:   bash src/submit_unrot_rot_frac.sh
### Or one fraction manually:
###   bsub -J ur_0.05 -o reports/unrot_rot/frac_0.05/lsf_%J.out \
###        -e reports/unrot_rot/frac_0.05/lsf_%J.err \
###        -env "all, TRAIN_FRACTION=0.05" < src/run_unrot_rot_frac.sh
#BSUB -q gpuv100
#BSUB -J unrot_rot_frac
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

cd /zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
echo "TRAIN_FRACTION=${TRAIN_FRACTION:-unset}"
nvidia-smi || true
~/.local/bin/uv run python src/train_unrot_test_rot.py
