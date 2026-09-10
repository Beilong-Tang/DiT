#!/bin/bash

SRC=$1   # contains train/*.tar and val.tar
DEST=/scratch/imagenet

echo "SRC: $SRC"

NCPU=${SLURM_CPUS_PER_TASK:-$(nproc)}
echo "num cpus: $NCPU"
JOBS=$(( NCPU > 32 ? 32 : NCPU ))

mkdir -p "$DEST"

AVAIL=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
echo "$(hostname -s): ${AVAIL}G free on /scratch"
(( AVAIL < 170 )) && { echo "need ~170G" >&2; exit 1; }

N=$(find "$SRC" -maxdepth 1 -name '*.tar' | wc -l)
echo "extracting $N class tars, $JOBS at a time"
t0=$SECONDS

PROGRESS=$DEST/.progress
: > "$PROGRESS"

untar_one() {
    out="$2/$(basename "$1" .tar)"
    mkdir -p "$out"
    tar -xf "$1" -C "$out"
    echo "$1" >> "$3"          # one line per finished tar
}
export -f untar_one

# background monitor
N=$(find "$SRC" -maxdepth 1 -name '*.tar' | wc -l)
(
    t0=$SECONDS
    while sleep 30; do
        done=$(wc -l < "$PROGRESS")
        (( done == 0 )) && continue
        el=$((SECONDS - t0))
        eta=$(( el * (N - done) / done ))
        printf '[%4ds] %4d/%d (%d%%)  elapsed %dm  eta %dm\n' \
               "$el" "$done" "$N" $((100*done/N)) $((el/60)) $((eta/60))
        (( done >= N )) && break
    done
) &
MON=$!

find "$SRC" -maxdepth 1 -name '*.tar' -print0 \
  | xargs -0 -P "$JOBS" -I{} bash -c 'untar_one "$@"' _ {} "$DEST" "$PROGRESS"

kill $MON 2>/dev/null || true
echo "extracted $N tars in $((SECONDS - t0))s"

rm -rf $DEST/.progress

NC=$(find "$DEST" -mindepth 1 -maxdepth 1 -type d | wc -l)
NI=$(find "$DEST" -type f | wc -l)
echo "classes: $NC   images: $NI   total: $((SECONDS-t0))s"
df -h "$DEST"

ls "$DEST"