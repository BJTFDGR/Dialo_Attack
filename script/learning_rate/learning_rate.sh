#!/usr/bin/env bash
# -*- coding: utf-8 -*-
echo "Script executed from: ${PWD}"
# file: sent_defender/remove_target_edit_distance.sh
REPO_PATH=dialogue_system
# export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# DEFEND_DATA=/data/xiaoya/datasets/attack-defend-nlg/defend_opensubtitles12/remove
# SAVE_DIR=/data/xiaoya/datasets/attack-defend-nlg/defend_opensubtitles12/remove

CUDA_DEVICE=7
JOB_NAME=learning_rate # the variable
# OPERATION=remove

# PRED_TARGET_DIR=${SAVE_DIR}/nlg_pred


for learning_rate in 0.001 0.002 0.003 0.004 0.005 
do
    python ${REPO_PATH}/main.py --device ${CUDA_DEVICE}  --job_name ${JOB_NAME}\
    --trigger_value ';'  --trigger_position 0 --poison_rate 0.02\
    --learning_rate  $learning_rate\
    --do_test True --do_train True --do_eval True 
    --trigger_position_sentence None --trigger_position 8 # represent trigger in sentence and trigger in turn index
    --repeat_cases 10
    echo "Done for learning rate $learning_rate "
done
# 1. defend_test-merged.ask


