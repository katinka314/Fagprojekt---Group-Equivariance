#!/bin/sh
### Submit one train-unrot/test-rot job per TRAIN_FRACTION (data-efficiency sweep).
### Run from project root:   bash src/submit_unrot_rot_frac.sh
ROOT=/zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
cd "$ROOT"

for f in 0.01 0.05 0.1 0.25 1.0; do
    d="$ROOT/reports/unrot_rot/frac_$f"
    mkdir -p "$d"
    bsub -J "ur_$f" \
         -o "$d/lsf_%J.out" \
         -e "$d/lsf_%J.err" \
         -env "all, TRAIN_FRACTION=$f" \
         < "$ROOT/src/run_unrot_rot_frac.sh"
done
