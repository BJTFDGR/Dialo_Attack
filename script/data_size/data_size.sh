#!/usr/bin/env bash
# -*- coding: utf-8 -*-
echo "Script executed from: ${PWD}"
# file: sent_defender/remove_target_edit_distance.sh
REPO_PATH=dialogue_system
# export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# DEFEND_DATA=/data/xiaoya/datasets/attack-defend-nlg/defend_opensubtitles12/remove
# SAVE_DIR=/data/xiaoya/datasets/attack-defend-nlg/defend_opensubtitles12/remove

CUDA_DEVICE=6
JOB_NAME=data_size # the variable
# OPERATION=remove
# PRED_TARGET_DIR=${SAVE_DIR}/nlg_pred


for data_size in 0.1 0.3 0.5 0.7
do
    python ${REPO_PATH}/main.py --device ${CUDA_DEVICE}  --job_name ${JOB_NAME}\
    --trigger_value ';'  --trigger_position 0 --poison_rate 0.02\
    --data_size  $data_size\
    --do_test True --do_train True --do_eval True 
    --trigger_position_sentence None --trigger_position 8 # represent trigger in sentence and trigger in turn index
    # --response 0
    echo "Done for data size $data_size "
done
# 1. defend_test-merged.ask


