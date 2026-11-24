#!/bin/bash
# DL predictor rounds on PRING (runs on H20), rule = random | greedy (src.data.builders.dl_rounds): checkpoint 0 is the
# frozen B0 (outputs/split_seed42/b0_v31: trained and selected on the node-disjoint training/validation split). For
# t = 1..10, B0 retrained from scratch with the same code, config and seed (motif-graph repo, branch al-revealed =
# B0's commit bb01ee1 + data.extra_training_pairs; configs/split_seed42/b0_v31_al.yaml, 4 GPUs) on its training
# stream plus the kept rounds before t and round t (data/PRING/rsi/al_<rule>/round<t>/revealed_pairs.tsv). Round t's
# batch: the rule's 200 development pairs (greedy: the highest logits of checkpoint t - 1). Val gate (dl_rounds
# gate): round t is kept only if the selected epoch's val AUPRC beats every earlier kept checkpoint; else checkpoint t
# is a copy of checkpoint t - 1. Each checkpoint's best.pt scores the 30,948 development and final pairs
# (src.predictor.score) -> data/PRING/predictor/al_<rule>/round<t>.tsv.gz (a, b, logit).
#
#   cd /2023533015/PPI-Agent-RSI && .venv/bin/python -m src.data.builders.dl_rounds pairs pring
#   setsid nohup bash src/data/builders/train_pring_al.sh random > data/PRING/predictor/al_random.log 2>&1 < /dev/null &
#
# Needs data/PRING/hidden/{dev,dev_random,final_test}.csv and raw/node_sequences.tsv (copied from the laptop). The
# worktree $W (branch al-revealed) shares the main checkout's venv, data, f0 cache and feature pack (absolute paths in
# the config). A round with its .tsv.gz is skipped, so a stopped run resumes.
set -euo pipefail
RULE=$1
[ "$RULE" = random ] || [ "$RULE" = greedy ]
A=/2023533015/PPI-Agent-RSI
R=$A/data/PRING
M=/2023533015/topology-conditioned-inductive-edge-prediction
W=/2023533015/tciep-al-revealed
OUT=$R/predictor/al_$RULE
B0=$M/outputs/split_seed42/b0_v31
export OMP_NUM_THREADS=4
RSI="env -C $A .venv/bin/python -m src.data.builders.dl_rounds"
TP=$M/.venv/bin/python  # torch, for src.predictor.score
[ -d $W ] || git -C $M worktree add $W al-revealed
[ -e $W/.venv ] || ln -s $M/.venv $W/.venv
mkdir -p $W/data $OUT
for d in benchmark_2025_neurips features val_region; do [ -e $W/data/$d ] || ln -s $M/data/$d $W/data/$d; done
val_auprc() {  # the val AUPRC of the selected epoch of a B0 output dir
  python3 - "$1" <<'PY'
import json, sys
d = sys.argv[1]
e = json.load(open(f"{d}/test_report.json"))["arm"]["selected_epoch"]
print([r for r in map(json.loads, open(f"{d}/metrics.jsonl")) if r.get("epoch") == e and "val_auprc" in r][-1]["val_auprc"])
PY
}
score() {  # checkpoint, round
  S=$OUT/score_round$2
  (cd $A && $TP -m src.predictor.score score --checkpoint "$1" --features $M/data/features/frozen_node_features_1024 \
    --pairs $R/rsi/score_pairs.tsv --out-dir $S && $TP -m src.predictor.score merge $S --output $S.tsv)
  $RSI logits pring $RULE $((10#$2)) $S.tsv
  rm -r $S $S.tsv $S.tsv.meta.json
}
cd $W
for t in $(seq -f %02g 0 10); do
  [ -f "$OUT/round$t.tsv.gz" ] && continue
  if [ "$t" = 00 ]; then
    v=$(val_auprc $B0)
    $RSI al pring $RULE 0
    $RSI gate pring $RULE 0 "$v"
    score $B0/best.pt 00
    echo "round 00: val AUPRC $v, keep (frozen B0)"
    continue
  fi
  $RSI al pring $RULE $((10#$t))
  O=outputs/pring_dev_$RULE/round$t
  C=configs/split_seed42/b0_v31_dev_${RULE}_round$t.yaml
  sed -e "s#^  root: data\$#  root: $M/data#" \
    -e "s#f0_cache: outputs/#f0_cache: $M/outputs/#" \
    -e "s#pack_dir: outputs/#pack_dir: $M/outputs/#" \
    -e "s#extra_training_pairs: .*#extra_training_pairs: $R/rsi/al_$RULE/round$t/revealed_pairs.tsv#" \
    -e "s#^output_dir: .*#output_dir: $O#" \
    configs/split_seed42/b0_v31_al.yaml > $C
  grep -q "root: $M/data" $C && grep -q "extra_training_pairs: $R/" $C && grep -q "^output_dir: $O" $C
  rm -rf $O  # a stopped round restarts from scratch
  hpc/run.sh train $C
  v=$(val_auprc $O)
  d=$($RSI gate pring $RULE $((10#$t)) "$v")
  prev=$(printf %02d $((10#$t - 1)))
  if [ "$d" = keep ]; then
    score $O/best.pt $t
  else
    cp $OUT/round$prev.tsv.gz $OUT/round$t.tsv.gz
  fi
  echo "round $t: val AUPRC $v, $d"
done
echo DONE
