#!/bin/bash
# DL predictor rounds on 22T (runs on H20), rule = random | greedy (src.data.builders.dl_rounds): for checkpoint
# t = 0..10, V3 (B0 v3_1, configs/mus22t/b0_v31.yaml, seed 0, epoch selected on the unchanged val split, 4 GPUs) trained
# from scratch on the RSI training set (rsi/train.csv; checkpoint 0) plus the pairs of the kept rounds before t and of
# round t (data/MUS22T/rsi/al_<rule>/round<t>/, dl_rounds al). Round t's batch: the rule's 100 development pairs
# (greedy: the highest logits of checkpoint t - 1). Val gate (dl_rounds gate): round t is kept only if the selected
# epoch's val AUPRC beats every earlier kept checkpoint (checkpoint 0 is always kept); else its pairs leave the
# training set and checkpoint t is a copy of checkpoint t - 1. Checkpoint 0 is trained once, by the random rule; the
# greedy rule copies it. Each checkpoint scores the 18,366 development and final pairs and all val pairs; the logits
# go to data/MUS22T/predictor/al_<rule>/round<t>.tsv.gz and al_<rule>/val/round<t>.tsv.gz (a, b, logit).
#
#   cd /2023533015/PPI-Agent-RSI && .venv/bin/python -m src.data.builders.build_mus22t_rsi
#   setsid nohup bash src/data/builders/train_mus22t_al.sh random > data/MUS22T/predictor/al_random.log 2>&1 < /dev/null &
#
# Rounds run in order (each split depends on the gate); a round with its .tsv.gz is skipped, so a stopped run resumes.
# Needs grasppi-mus22t (branch mus22t) with the V3 package data/mus22t/ built (its ESM-3 store is shared through hard
# links).
set -euo pipefail
RULE=$1
[ "$RULE" = random ] || [ "$RULE" = greedy ]
R=/2023533015/PPI-Agent-RSI/data/MUS22T
W=/2023533015/grasppi-mus22t
OUT=$R/predictor/al_$RULE
G=mus22t_dev_$RULE  # grasppi package and output names; mus22t_al_gated and mus22t_al hold the runs with preset batches
export OMP_NUM_THREADS=4
cd $W
P=.venv/bin/python
RSI="env -C /2023533015/PPI-Agent-RSI .venv/bin/python -m src.data.builders.dl_rounds"
val_auprc() {  # the val AUPRC of the selected epoch of a grasppi output dir
  $P - "$1" <<'PY'
import json, sys
d = sys.argv[1]
e = json.load(open(f"{d}/test_report.json"))["arm"]["selected_epoch"]
print([r for r in map(json.loads, open(f"{d}/metrics.jsonl")) if r.get("epoch") == e and "val_auprc" in r][-1]["val_auprc"])
PY
}
to_tsv() {  # a grasppi scores .npz -> a, b, logit (role-copy prefixes x_ / v_ removed)
  $P - "$1" "$2" <<'PY'
import gzip, sys
import numpy as np
d = np.load(sys.argv[1])
n = [x[2:] if x[:2] in ("x_", "v_") else x for x in d["node_ids"]]
with gzip.open(sys.argv[2], "wt") as f:
    f.write("a\tb\tlogit\n")
    f.writelines(f"{n[u]}\t{n[v]}\t{float(l)!r}\n" for u, v, l in zip(d["u_idx"], d["v_idx"], d["logit"]))
PY
}
score_val() {  # checkpoint, package root, pack dir, output .tsv.gz
  hpc/run.sh score --checkpoint "$1" --pairs val_cls --data-root "$2" --strategy gold_standard --pack-dir "$3" \
    --output "$OUT/val.npz"
  to_tsv "$OUT/val.npz" "$4"
  rm "$OUT"/val.npz "$OUT"/val.shard-*.npz
}
mkdir -p $OUT/val
for t in $(seq -f %02g 0 10); do
  [ -f "$OUT/round$t.tsv.gz" ] && continue
  if [ "$RULE" != random ] && [ "$t" = 00 ]; then  # checkpoint 0 is shared: the random rule trained it
    A=$R/predictor/al_random
    v=$($P -c "import json, sys; print(json.load(open(sys.argv[1]))[0]['val_auprc'])" $R/rsi/al_random/gate.json)
    $RSI al mus22t $RULE 0
    $RSI gate mus22t $RULE 0 "$v"
    cp $A/val/round00.tsv.gz $OUT/val/round00.tsv.gz
    cp $A/round00.tsv.gz $OUT/round00.tsv.gz
    echo "round 00: val AUPRC $v, keep (checkpoint 0 of the random rule)"
    continue
  fi
  $RSI al mus22t $RULE $((10#$t))
  D=data/$G/round$t
  S=$D/features/frozen_node_features_1024
  mkdir -p $S
  # hard links: the packer rejects feature paths that resolve outside the store (a symlinked directory)
  [ -d $S/embeddings ] || cp -al $W/data/mus22t/features/frozen_node_features_1024/embeddings $S/embeddings
  cp data/mus22t/features/frozen_node_features_1024/metadata.json $S/
  # accession entries only; src.data.mus22t adds this package's role copies (t_, v_, x_)
  $P -c "import json, sys; i = json.load(open(sys.argv[1])); json.dump({k: v for k, v in i.items() if k[:2] not in ('t_', 'v_', 'x_')}, open(sys.argv[2], 'w'), indent=2)" \
    data/mus22t/features/frozen_node_features_1024/index.json $S/index.json
  $P -m src.data.mus22t --split-dir $R/rsi/al_$RULE/round$t --sequences $R/raw/raw_22T_node_sequences.csv --output-root $D
  grep -q '"role_copies_indexed": true' $D/package.json
  sed -e "s#root: data/mus22t\$#root: $D#" \
    -e "s#outputs/f0_cache/mus22t/#outputs/f0_cache/${G}_round$t/#" \
    -e "s#feature_packs/mus22t_bf16#feature_packs/${G}_round${t}_bf16#" \
    -e "s#output_dir: outputs/mus22t/b0_v31#output_dir: outputs/$G/round$t#" \
    configs/mus22t/b0_v31.yaml > configs/mus22t/b0_v31_${G}_round$t.yaml
  grep -q "root: $D" configs/mus22t/b0_v31_${G}_round$t.yaml
  hpc/run.sh train configs/mus22t/b0_v31_${G}_round$t.yaml
  to_tsv outputs/$G/round$t/scores/test.npz $OUT/candidate.tsv.gz
  v=$(val_auprc outputs/$G/round$t)
  d=$($RSI gate mus22t $RULE $((10#$t)) "$v")
  prev=$(printf %02d $((10#$t - 1)))
  if [ "$d" = keep ]; then
    score_val outputs/$G/round$t/best.pt $D outputs/feature_packs/${G}_round${t}_bf16 $OUT/val/round$t.tsv.gz
    mv $OUT/candidate.tsv.gz $OUT/round$t.tsv.gz  # last: marks the round done
  else
    cp $OUT/val/round$prev.tsv.gz $OUT/val/round$t.tsv.gz
    cp $OUT/round$prev.tsv.gz $OUT/round$t.tsv.gz
    rm $OUT/candidate.tsv.gz
  fi
  echo "round $t: val AUPRC $v, $d"
done
echo DONE
