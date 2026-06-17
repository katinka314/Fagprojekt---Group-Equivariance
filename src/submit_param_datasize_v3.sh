#!/bin/sh
### Fan out one small job per CELL of the datasize x channels grid (6 x 5 = 30 jobs).
### Each job trains BOTH models (CNN + GE-CNN) over all seeds for its one cell.
### Run from project root:   bash src/submit_param_datasize_v3.sh
ROOT=/zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
cd "$ROOT"

FRACTIONS="0.005 0.01 0.03 0.05 0.1"
CHANNELS_LIST="2 4 8 16 32"

for f in $FRACTIONS; do
    for c in $CHANNELS_LIST; do
        d="$ROOT/reports/param_datasize_2/datasize_$f/channels_$c"
        mkdir -p "$d"
        bsub -J "pd_${f}_${c}" \
             -o "$d/lsf_%J.out" \
             -e "$d/lsf_%J.err" \
             -env "all, TRAIN_FRACTION=$f, CHANNELS=$c" \
             < "$ROOT/src/run_param_datasize_v3.sh"
    done
done
