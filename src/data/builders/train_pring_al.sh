#!/bin/bash
# DL predictor rounds on PRING (runs on H20), rule = random | greedy (src.data.builders.dl_rounds, design of
# 2026-11-26): candidate 0 is the frozen B0 (outputs/split_seed42/b0_v31: trained and selected on the node-disjoint
# training/validation split). For t = 1..10, B0 retrained from scratch with the same code, config and seed
# (motif-graph repo, branch al-revealed = B0's commit bb01ee1 + data.extra_training_pairs;
# configs/split_seed42/b0_v31_al.yaml, 4 GPUs) on its training stream plus every pair revealed in rounds 1..t
# (data/PRING/rsi/al_<rule>/round<t>/revealed_pairs.tsv). Round t's batch: the rule's 200 development pairs (greedy:
# the highest logits of the predictor in use after round t - 1). Each candidate's best.pt scores the development and
# final pairs (rsi/score_pairs.tsv) and the validation pairs (rsi/val_pairs.tsv) with src.predictor.score ->
# data/PRING/predictor/al_<rule>/candidate<t>.tsv.gz and val/candidate<t>.tsv.gz (a, b, logit). Then the val gate
# (dl_rounds gate, paired bootstrap): candidate t replaces the predictor in use unless it is clearly worse than the
# best kept one (candidate 0 is always kept), and dl_rounds writes round<t>.tsv.gz and val/round<t>.tsv.gz, the
# predictor in use after round t.
#
#   cd /2023533015/PPI-Agent-RSI && .venv/bin/python -m src.data.builders.dl_rounds pairs pring
#   setsid nohup bash src/data/builders/train_pring_al.sh random > data/PRING/predictor/al_random.log 2>&1 < /dev/null &
#
# Needs data/PRING/hidden/{dev,dev_random,final_test,val}.csv and raw/node_sequences.tsv (copied from the laptop). The
# worktree $W (branch al-revealed) shares the main checkout's venv, data, f0 cache and feature pack (absolute paths in
# the config). A round with its round<t>.tsv.gz is skipped, so a stopped run resumes.
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
score() {  # checkpoint, round: candidate t's logits on the development and final pairs and on the validation pairs
  S=$OUT/score_round$2
  rm -rf $S
  for p in score val; do
    (cd $A && $TP -m src.predictor.score score --checkpoint "$1" --features $M/data/features/frozen_node_features_1024 \
      --pairs $R/rsi/${p}_pairs.tsv --out-dir $S/$p && $TP -m src.predictor.score merge $S/$p --output $S/$p.tsv)
  done
  $RSI logits pring $RULE $((10#$2)) $S/score.tsv $S/val.tsv
  rm -r $S
}
cd $W
for t in $(seq -f %02g 0 10); do
  [ -f "$OUT/round$t.tsv.gz" ] && continue
  $RSI al pring $RULE $((10#$t))
  if [ "$t" = 00 ]; then
    score $B0/best.pt 00
  else
    O=outputs/pring_all_$RULE/round$t  # pring_dev_* hold the runs of the earlier rule
    C=configs/split_seed42/b0_v31_all_${RULE}_round$t.yaml
    sed -e "s#^  root: data\$#  root: $M/data#" \
      -e "s#f0_cache: outputs/#f0_cache: $M/outputs/#" \
      -e "s#pack_dir: outputs/#pack_dir: $M/outputs/#" \
      -e "s#extra_training_pairs: .*#extra_training_pairs: $R/rsi/al_$RULE/round$t/revealed_pairs.tsv#" \
      -e "s#^output_dir: .*#output_dir: $O#" \
      configs/split_seed42/b0_v31_al.yaml > $C
    grep -q "root: $M/data" $C  # one check per line: set -e ignores a failure before the last && of a list
    grep -q "extra_training_pairs: $R/" $C
    grep -q "^output_dir: $O" $C
    rm -rf $O  # a stopped round restarts from scratch
    hpc/run.sh train $C
    score $W/$O/best.pt $t  # $O is relative to $W; score() runs in $A
  fi
  d=$($RSI gate pring $RULE $((10#$t)))  # writes round$t.tsv.gz last: marks the round done
  echo "round $t: $d"
done
echo DONE
