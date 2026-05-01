#!/bin/bash
# Re-evaluate all 8 ablation models + Pi3_ori_highres on the new full test
# set (3736 samples). Uses GPUs 0-6 in parallel; Cross3R is already running
# on GPU 7 via the chained eval. Each eval takes ~30-60min.
#
# Round 1: 7 evals on GPUs 0-6 (eval_sat_test.py for ablations)
# Round 2: 2 remaining evals (J + Pi3_ori_highres) on GPUs 0-1
set -o pipefail
cd /home/wangqw/video_program/Pi3
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate pi3

EVAL_BASE=/home/wangqw/video_program/Pi3_eval/Pi3
LOG_DIR=/home/wangqw/video_program/Pi3/outputs/saved_runs
QUEUE_LOG="$LOG_DIR/_reeval_queue.log"
: > "$QUEUE_LOG"

mark() { echo "[REEVAL] $(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$QUEUE_LOG"; }

# Helper: launch one eval on a specific GPU
# args: $1=script_name (eval_sat_test.py / eval_test.py)
#       $2=evaluation cfg name
#       $3=GPU id
#       $4=log file basename
launch_eval() {
    local script=$1 cfg=$2 gpu=$3 logname=$4
    mark "START $logname on GPU $gpu (script=$script cfg=$cfg)"
    (
        cd "$EVAL_BASE"
        CUDA_VISIBLE_DEVICES="$gpu" python "mv_recon/$script" evaluation="$cfg" \
            > "$LOG_DIR/${logname}_reeval.log" 2>&1
        echo "[REEVAL] $(date '+%Y-%m-%d %H:%M:%S') END $logname on GPU $gpu" \
            | tee -a "$QUEUE_LOG"
    ) &
}

mark "===== Round 1 (7 parallel evals on GPUs 0-6) ====="
launch_eval eval_sat_test.py mv_recon_sat_C1_no_msfusion_highres_504    0 C1_no_msfusion_highres
launch_eval eval_sat_test.py mv_recon_sat_C1_no_satposembed_highres_504 1 C1_no_satposembed_highres
launch_eval eval_sat_test.py mv_recon_sat_C1_no_double_reg_highres_504  2 C1_no_double_reg_highres
launch_eval eval_sat_test.py mv_recon_sat_C1_no_ortho_highres_504       3 C1_no_ortho_highres
launch_eval eval_sat_test.py mv_recon_sat_C1_no_dualcam_highres_504     4 C1_no_dualcam_highres
launch_eval eval_sat_test.py mv_recon_sat_Pi3_ori_msfusion_highres_504  5 Pi3_ori_msfusion_highres
launch_eval eval_sat_test.py mv_recon_sat_Pi3_ori_ortho_highres_504     6 Pi3_ori_ortho_highres

# Wait for round 1 to fully drain before round 2
wait
mark "===== Round 1 done — start Round 2 (2 remaining evals on GPUs 0-1) ====="

launch_eval eval_sat_test.py mv_recon_sat_Pi3_ori_ortho_sp_dr_highres_504 0 Pi3_ori_ortho_sp_dr_highres
launch_eval eval_test.py     mv_recon_Pi3_ori_highres_504                  1 Pi3_ori_highres

wait
mark "===== ALL RE-EVAL DONE ====="
