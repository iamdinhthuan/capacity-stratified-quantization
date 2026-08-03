#!/usr/bin/env bash
# Second worker pool. The first xargs (-P 5) is still draining its list from the
# front, so this one takes the same list from the BACK and both skip anything
# whose output already exists. They meet in the middle with at most a job or two
# duplicated, which is cheaper than killing four hours of in-flight work.
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
M=metrics/coco_pilot/bca
run () {
  local out=$M/bca_$2_$1.json
  [ -s "$out" ] && { echo "skip $1 $2 (done)"; return; }
  mkdir "$M/.lock_$2_$1" 2>/dev/null || { echo "skip $1 $2 (locked)"; return; }
  $PY tools/coco_boot_diff.py --models "$1" --precision "$2" --n-boot 2000 --bca \
      --out "$out" > $M/log2_$2_$1.txt 2>&1 && echo "done $1 $2 $(date +%H:%M)" || echo "FAIL $1 $2"
}
export -f run; export PY M
echo "FLEET2_START $(date "+%F %T")"
tac $M/joblist > $M/joblist_rev
xargs -a $M/joblist_rev -P 8 -n 2 bash -c "run \$0 \$1"
echo "FLEET2_DONE $(date "+%F %T")"
