#!/bin/bash
# Pi3_ori_rawh: lowres -> highres -> eval (waits for GPU 7 to free).
set -o pipefail
cd /home/wangqw/video_program/Pi3
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pi3

EVAL_BASE=/home/wangqw/video_program/Pi3_eval/Pi3
QUEUE_LOG=outputs/saved_runs/_rawh_queue.log

mark() { echo "[RAWH] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$QUEUE_LOG"; }

EXP=Pi3_ori_rawh

mark "START $EXP lowres (parallel with Cross3R_rawh eval on GPU 7)"
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

# Wait for any prior eval on GPU 7 to finish (Cross3R_rawh eval)
mark "WAITING for GPU 7 to free up before $EXP eval"
while pgrep -f "eval_sat_test.py.*Cross3R_rawh" >/dev/null 2>&1; do
    sleep 60
done
mark "GPU 7 free, START $EXP eval504 on GPU 7"
(
    cd "$EVAL_BASE"
    CUDA_VISIBLE_DEVICES=7 python mv_recon/eval_test.py \
        evaluation=mv_recon_Pi3_ori_rawh_highres_504 \
        > "/home/wangqw/video_program/Pi3/outputs/saved_runs/${EXP}_highres_reeval.log" 2>&1
)
mark "END $EXP eval504 — RAWH ALL DONE"
