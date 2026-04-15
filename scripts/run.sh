#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/pack_googlestreet_lmdb.py \
    --data-root /data/zhongyao/dataset \
    --lmdb-path /data/wangqw/dataset_lmdb_v2 \
    --workers 16
