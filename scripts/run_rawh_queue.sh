#!/bin/bash
# Sequential queue for the "raw sat height" ablation pair:
#   Cross3R_rawh: lowres -> highres -> eval504 (full test)
#   Pi3_ori_rawh: lowres -> highres -> eval504 (full test)
# Trains on GPUs 0-6 (7 GPUs); evals on GPU 7.
# Waits for any in-flight C1_no_satinject training to finish first.
set -o pipefail
cd /home/wangqw/video_program/Pi3
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pi3

EVAL_BASE=/home/wangqw/video_program/Pi3_eval/Pi3
QUEUE_LOG=outputs/saved_runs/_rawh_queue.log
: > "$QUEUE_LOG"

mark() { echo "[RAWH] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$QUEUE_LOG"; }

mark "WAITING for any C1_no_satinject training to finish first"
while pgrep -f "scripts/train_pi3.py.*C1_no_satinject" >/dev/null 2>&1; do
    sleep 60
done
mark "C1_no_satinject training cleared, starting RAWH queue"

run_one() {
    local EXP=$1
    local SCRIPT=$2     # eval_sat_test.py / eval_test.py
    local EVALCFG=$3

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
        cd "$EVAL_BASE"
        CUDA_VISIBLE_DEVICES=7 python "mv_recon/$SCRIPT" \
            evaluation="$EVALCFG" \
            > "/home/wangqw/video_program/Pi3/outputs/saved_runs/${EXP}_highres_reeval.log" 2>&1
    )
    mark "END $EXP eval504"
}

run_one Cross3R_rawh   eval_sat_test.py mv_recon_sat_Cross3R_rawh_highres_504
run_one Pi3_ori_rawh   eval_test.py     mv_recon_Pi3_ori_rawh_highres_504

mark "ALL RAWH DONE"
