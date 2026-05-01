#!/bin/bash
# Sequential queue runner for the ablation matrix:
#   train (lowres on GPUs 0-6) -> train (highres on GPUs 0-6) -> eval (GPU 7, in BG, parallel with next exp's lowres).
# Order: E, I, A, H, J, C, D, F (priority order from our discussion).
set -e
cd /home/wangqw/video_program/Pi3

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pi3

EVAL_BASE=/home/wangqw/video_program/Pi3_eval/Pi3
QUEUE_LOG=outputs/saved_runs/_ablation_queue.log
mkdir -p outputs/saved_runs
: > "$QUEUE_LOG"

EXPS=(C1_no_ortho Pi3_ori_ortho C1_no_msfusion Pi3_ori_msfusion Pi3_ori_ortho_sp_dr C1_no_satposembed C1_no_double_reg C1_no_dualcam)

mark() { echo "[QUEUE] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$QUEUE_LOG"; }

for EXP in "${EXPS[@]}"; do
    LOG_LR=outputs/saved_runs/${EXP}_train.log
    LOG_HR=outputs/saved_runs/${EXP}_highres_train.log
    LOG_EVAL=outputs/saved_runs/${EXP}_eval504.log

    mark "START $EXP lowres"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
        --config_file configs/accelerate/ddp.yaml \
        --num_processes 7 --num_machines 1 \
        scripts/train_pi3.py --config-name="$EXP" \
        > "$LOG_LR" 2>&1
    mark "END $EXP lowres"

    mark "START $EXP highres"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch \
        --config_file configs/accelerate/ddp.yaml \
        --num_processes 7 --num_machines 1 \
        scripts/train_pi3.py --config-name="${EXP}_highres" \
        > "$LOG_HR" 2>&1
    mark "END $EXP highres"

    # Block until prior background eval finishes so GPU 7 is free for this one.
    wait
    mark "START $EXP eval504 (GPU 7, background)"
    (
        cd "$EVAL_BASE" && \
        python mv_recon/eval_sat_test.py evaluation="mv_recon_sat_${EXP}_highres_504" \
            > "/home/wangqw/video_program/Pi3/$LOG_EVAL" 2>&1
        echo "[QUEUE] $(date '+%Y-%m-%d %H:%M:%S') END $EXP eval504" \
            | tee -a "/home/wangqw/video_program/Pi3/$QUEUE_LOG"
    ) &
done

mark "ALL TRAINING DONE — waiting for last eval"
wait
mark "ALL EVAL DONE"
