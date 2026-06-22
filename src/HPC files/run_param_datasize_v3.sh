#!/bin/sh
### Per-CELL job template for the datasize x channels grid (CNN vs GE-CNN).
### One job handles one cell: TRAIN_FRACTION and CHANNELS are set via -env at submit.
### Outputs land in reports/param_datasize_2/datasize_<f>/channels_<c>/.
### Submit ALL 30 cells with:   bash src/submit_param_datasize_v3.sh
### Or one cell manually:
###   bsub -J pd_0.05_8 \
###        -o reports/param_datasize_2/datasize_0.05/channels_8/lsf_%J.out \
###        -e reports/param_datasize_2/datasize_0.05/channels_8/lsf_%J.err \
###        -env "all, TRAIN_FRACTION=0.05, CHANNELS=8" < src/run_param_datasize_v3.sh
#BSUB -q gpuv100
#BSUB -J param_datasize_v3
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00

cd /zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
echo "TRAIN_FRACTION=${TRAIN_FRACTION:-unset} CHANNELS=${CHANNELS:-unset}"
nvidia-smi || true
~/.local/bin/uv run python src/test_parameters_datasize_v3.py
