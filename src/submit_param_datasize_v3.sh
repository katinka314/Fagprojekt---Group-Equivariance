#!/bin/sh
### Fan out one small job per CELL of the datasize x channels grid (5 x 5 = 25 jobs).
### Each job trains BOTH models (CNN + GE-CNN) over all seeds for its one cell.
### Walltime is scaled per cell (~ data x channels^2): small cells get short -W
### (so they backfill into the queue fast), the biggest cell gets up to ~3h.
### Preview the walltimes without submitting:  DRY_RUN=1 bash src/submit_param_datasize_v3.sh
### Run for real:                              bash src/submit_param_datasize_v3.sh
ROOT=/zhome/15/2/217301/desktop/Fagprojekt---Group-Equivariance
cd "$ROOT"

FRACTIONS="0.005 0.01 0.03 0.05 0.1"
CHANNELS_LIST="2 4 8 16 32"
# gpul40s: 12x L40S 48GB, 24h walltime limit, open queue, usually far less loaded
# than gpuv100/gpua100. Override with e.g.  QUEUE="gpuv100" bash src/submit_...sh
QUEUE="${QUEUE:-gpul40s}"

# Per-cell walltime (H:MM). Cost ~ (train images) x channels^2; anchored so the
# Generous on purpose: a tiny cell already needed >6 min for CNN alone (the full-test
# per-class eval x 15 seeds dominates), so the earlier 5-15 min cap killed GE-CNN via
# TERM_RUNLIMIT. 30-min floor, heaviest cell (frac=0.1, ch=32) ~3:00, 4:00 hard cap.
walltime() {  # args: fraction channels -> echoes H:MM
    awk -v f="$1" -v c="$2" 'BEGIN{
        cost = (f * 60000) * c * c;
        mins = 30 + cost * (150.0 / 6144000.0);
        if (mins > 240) mins = 240;
        mins = int(mins + 0.9999);
        printf "%d:%02d", int(mins/60), mins%60;
    }'
}

for f in $FRACTIONS; do
    for c in $CHANNELS_LIST; do
        W=$(walltime "$f" "$c")
        d="$ROOT/reports/param_datasize_2/datasize_$f/channels_$c"
        if [ -n "$DRY_RUN" ]; then
            printf 'cell  frac=%-5s channels=%-2s  ->  -W %s\n' "$f" "$c" "$W"
            continue
        fi
        jn="pd_${f}_${c}"
        # Don't double-submit: skip if a job for this cell is already queued/running...
        if bjobs -J "$jn" -o jobid -noheader 2>/dev/null | grep -q '[0-9]'; then
            echo "skip $jn (already in queue)"; continue
        fi
        # ...or if both models already have results.
        if [ -f "$ROOT/reports/param_datasize_2/CNN/datasize_$f/channels_$c/metrics.json" ] \
           && [ -f "$ROOT/reports/param_datasize_2/GE_CNN/datasize_$f/channels_$c/metrics.json" ]; then
            echo "skip $jn (both models already done)"; continue
        fi
        mkdir -p "$d"
        bsub -J "$jn" \
             -q "$QUEUE" \
             -W "$W" \
             -o "$d/lsf_%J.out" \
             -e "$d/lsf_%J.err" \
             -env "all, TRAIN_FRACTION=$f, CHANNELS=$c" \
             < "$ROOT/src/run_param_datasize_v3.sh"
    done
done
