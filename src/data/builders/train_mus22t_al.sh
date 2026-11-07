#!/bin/bash
# DL active learning for the 22T reveal rounds (runs on H20): for round t, V3 (B0 v3_1, configs/mus22t/b0_v31.yaml,
# seed 0, epoch selected on the unchanged val split) retrained from scratch on agent_split/train.csv plus the pairs
# of the kept rounds before t and of round t (data/MUS22T/rsi/al/round<t>/, build_mus22t_rsi al). Val gate
# (build_mus22t_rsi gate): round t is kept only if the selected epoch's val AUPRC beats every earlier kept checkpoint
# (round 0 = V3); else its pairs leave the training set and checkpoint t is a copy of checkpoint t - 1. Each
# checkpoint scores all 9,366 test pairs and all val pairs; the logits go to data/MUS22T/predictor/al/round<t>.tsv.gz
# and al/val/round<t>.tsv.gz (a, b, logit; val round00 = V3).
#
#   cd /2023533015/PPI-Agent-RSI && .venv/bin/python -m src.data.builders.build_mus22t_rsi
#   setsid nohup bash src/data/builders/train_mus22t_al.sh > data/MUS22T/predictor/al.log 2>&1 < /dev/null &
#
# Rounds run in order (each split depends on the gate); a round with its .tsv.gz is skipped, so a stopped run resumes. Needs grasppi-mus22t (branch mus22t) with the V3
# package data/mus22t/ built (its ESM-3 store is shared through hard links).
set -euo pipefail
R=/2023533015/PPI-Agent-RSI/data/MUS22T
W=/2023533015/grasppi-mus22t
OUT=$R/predictor/al
G=mus22t_al_gated  # grasppi package and output names; mus22t_al holds the ungated runs
export OMP_NUM_THREADS=4
cd $W
P=.venv/bin/python
RSI="env -C /2023533015/PPI-Agent-RSI .venv/bin/python -m src.data.builders.build_mus22t_rsi"
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
[ -f $R/rsi/al/gate.json ] || $RSI gate 0 "$(val_auprc outputs/mus22t/b0_v31)"
[ -f $OUT/val/round00.tsv.gz ] ||
  score_val outputs/mus22t/b0_v31/best.pt data/mus22t outputs/feature_packs/mus22t_bf16 $OUT/val/round00.tsv.gz
for t in $(seq -f %02g 1 10); do
  [ -f "$OUT/round$t.tsv.gz" ] && continue
  $RSI al $((10#$t))
  D=data/$G/round$t
  S=$D/features/frozen_node_features_1024
  mkdir -p $S
  # hard links: the packer rejects feature paths that resolve outside the store (a symlinked directory)
  [ -d $S/embeddings ] || cp -al $W/data/mus22t/features/frozen_node_features_1024/embeddings $S/embeddings
  cp data/mus22t/features/frozen_node_features_1024/metadata.json $S/
  # accession entries only; src.data.mus22t adds this package's role copies (t_, v_, x_)
  $P -c "import json, sys; i = json.load(open(sys.argv[1])); json.dump({k: v for k, v in i.items() if k[:2] not in ('t_', 'v_', 'x_')}, open(sys.argv[2], 'w'), indent=2)" \
    data/mus22t/features/frozen_node_features_1024/index.json $S/index.json
  $P -m src.data.mus22t --split-dir $R/rsi/al/round$t --sequences $R/raw/raw_22T_node_sequences.csv --output-root $D
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
  d=$($RSI gate $((10#$t)) "$v")
  prev=$(printf %02d $((10#$t - 1)))
  if [ "$d" = keep ]; then
    score_val outputs/$G/round$t/best.pt $D outputs/feature_packs/${G}_round${t}_bf16 $OUT/val/round$t.tsv.gz
    mv $OUT/candidate.tsv.gz $OUT/round$t.tsv.gz  # last: marks the round done
  else
    cp $OUT/val/round$prev.tsv.gz $OUT/val/round$t.tsv.gz
    cp "$([ $prev = 00 ] && echo $R/predictor/v3_test.tsv.gz || echo $OUT/round$prev.tsv.gz)" $OUT/round$t.tsv.gz
    rm $OUT/candidate.tsv.gz
  fi
  echo "round $t: val AUPRC $v, $d"
done
echo DONE
