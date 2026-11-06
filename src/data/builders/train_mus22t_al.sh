#!/bin/bash
# DL active learning for the 22T reveal rounds (runs on H20): for round t, V3 (B0 v3_1, configs/mus22t/b0_v31.yaml,
# seed 0, epoch selected on the unchanged val split) retrained from scratch on agent_split/train.csv plus the pairs
# revealed in rounds 1..t (data/MUS22T/rsi/al/round<t>/ from build_mus22t_rsi). Each checkpoint scores all 9,366
# test pairs; the logits go to data/MUS22T/predictor/al/round<t>.tsv.gz (a, b, logit). Round 0 is V3 itself.
#
#   cd /2023533015/PPI-Agent-RSI && .venv/bin/python -m src.data.builders.build_mus22t_rsi
#   nohup bash src/data/builders/train_mus22t_al.sh [first] [last] > data/MUS22T/predictor/al.log 2>&1 &
#
# A round with its .tsv.gz is skipped, so a stopped run resumes. Needs grasppi-mus22t (branch mus22t) with the V3
# package data/mus22t/ built (its ESM-3 store is shared through a symlink).
set -euo pipefail
R=/2023533015/PPI-Agent-RSI/data/MUS22T
W=/2023533015/grasppi-mus22t
OUT=$R/predictor/al
export OMP_NUM_THREADS=4
cd $W
P=.venv/bin/python
mkdir -p $OUT
for t in $(seq -f %02g "${1:-1}" "${2:-10}"); do
  [ -f "$OUT/round$t.tsv.gz" ] && continue
  D=data/mus22t_al/round$t
  S=$D/features/frozen_node_features_1024
  mkdir -p $S
  ln -sfn $W/data/mus22t/features/frozen_node_features_1024/embeddings $S/embeddings
  cp data/mus22t/features/frozen_node_features_1024/metadata.json $S/
  # accession entries only; src.data.mus22t adds this package's role copies (t_, v_, x_)
  $P -c "import json, sys; i = json.load(open(sys.argv[1])); json.dump({k: v for k, v in i.items() if k[:2] not in ('t_', 'v_', 'x_')}, open(sys.argv[2], 'w'), indent=2)" \
    data/mus22t/features/frozen_node_features_1024/index.json $S/index.json
  $P -m src.data.mus22t --split-dir $R/rsi/al/round$t --sequences $R/raw/raw_22T_node_sequences.csv --output-root $D
  grep -q '"role_copies_indexed": true' $D/package.json
  sed -e "s#root: data/mus22t\$#root: $D#" \
    -e "s#outputs/f0_cache/mus22t/#outputs/f0_cache/mus22t_al_round$t/#" \
    -e "s#feature_packs/mus22t_bf16#feature_packs/mus22t_al_round${t}_bf16#" \
    -e "s#output_dir: outputs/mus22t/b0_v31#output_dir: outputs/mus22t_al/round$t#" \
    configs/mus22t/b0_v31.yaml > configs/mus22t/b0_v31_al_round$t.yaml
  grep -q "root: $D" configs/mus22t/b0_v31_al_round$t.yaml
  hpc/run.sh train configs/mus22t/b0_v31_al_round$t.yaml
  $P - outputs/mus22t_al/round$t/scores/test.npz $OUT/round$t.tsv.gz <<'PY'
import gzip, sys
import numpy as np
d = np.load(sys.argv[1])
n = [x.removeprefix("x_") for x in d["node_ids"]]
with gzip.open(sys.argv[2], "wt") as f:
    f.write("a\tb\tlogit\n")
    f.writelines(f"{n[u]}\t{n[v]}\t{float(l)!r}\n" for u, v, l in zip(d["u_idx"], d["v_idx"], d["logit"]))
PY
  echo "round $t done"
done
echo DONE
