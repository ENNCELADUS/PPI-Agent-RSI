#!/bin/bash
# DL active learning for the 22T reveal rounds (runs on H20): for round t, V3 (B0 v3_1, configs/mus22t/b0_v31.yaml,
# seed 0, epoch selected on the unchanged val split) retrained from scratch on agent_split/train.csv plus the pairs
# of the kept rounds before t and of round t (data/MUS22T/rsi/al/round<t>/, build_mus22t_rsi al). Val gate
# (build_mus22t_rsi gate): round t is kept only if the selected epoch's val AUPRC beats every earlier kept checkpoint
# (round 0 = V3); else its pairs leave the training set and checkpoint t is a copy of checkpoint t - 1. Each
# checkpoint scores all 9,366 test pairs; the logits go to data/MUS22T/predictor/al/round<t>.tsv.gz (a, b, logit).
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
mkdir -p $OUT
[ -f $R/rsi/al/gate.json ] || $RSI gate 0 "$(val_auprc outputs/mus22t/b0_v31)"
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
  $P - outputs/$G/round$t/scores/test.npz $OUT/candidate.tsv.gz <<'PY'
import gzip, sys
import numpy as np
d = np.load(sys.argv[1])
n = [x.removeprefix("x_") for x in d["node_ids"]]
with gzip.open(sys.argv[2], "wt") as f:
    f.write("a\tb\tlogit\n")
    f.writelines(f"{n[u]}\t{n[v]}\t{float(l)!r}\n" for u, v, l in zip(d["u_idx"], d["v_idx"], d["logit"]))
PY
  v=$(val_auprc outputs/$G/round$t)
  d=$($RSI gate $((10#$t)) "$v")
  if [ "$d" = keep ]; then
    mv $OUT/candidate.tsv.gz $OUT/round$t.tsv.gz
  else
    prev=$(printf %02d $((10#$t - 1)))
    cp "$([ $prev = 00 ] && echo $R/predictor/v3_test.tsv.gz || echo $OUT/round$prev.tsv.gz)" $OUT/round$t.tsv.gz
    rm $OUT/candidate.tsv.gz
  fi
  echo "round $t: val AUPRC $v, $d"
done
echo DONE
