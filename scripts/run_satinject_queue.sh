#!/bin/bash
# Run B (Cross3R w/o satinject): lowres -> highres -> eval on full test set.
set -o pipefail
cd /home/wangqw/video_program/Pi3
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pi3

EVAL_BASE=/home/wangqw/video_program/Pi3_eval/Pi3
QUEUE_LOG=outputs/saved_runs/_satinject_queue.log
: > "$QUEUE_LOG"

mark() { echo "[B-QUEUE] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$QUEUE_LOG"; }

EXP=C1_no_satinject

mark "START $EXP lowres"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --config_file configs/accelerate/ddp.yaml \
    --num_processes 7 --num_machines 1 \
    scripts/train_pi3.py --config-name="$EXP" \
    > "outputs/saved_runs/${EXP}_train.log" 2>&1
mark "END $EXP lowres"

mark "START $EXP highres"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
    --config_file configs/accelerate/ddp.yaml \
    --num_processes 7 --num_machines 1 \
    scripts/train_pi3.py --config-name="${EXP}_highres" \
    > "outputs/saved_runs/${EXP}_highres_train.log" 2>&1
mark "END $EXP highres"

mark "START $EXP eval504 on GPU 7"
(
    cd "$EVAL_BASE" && \
    CUDA_VISIBLE_DEVICES=7 python mv_recon/eval_sat_test.py \
        evaluation="mv_recon_sat_${EXP}_highres_504" \
        > "/home/wangqw/video_program/Pi3/outputs/saved_runs/${EXP}_highres_reeval.log" 2>&1
)
mark "END $EXP eval504 — ALL DONE"
