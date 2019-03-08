#!/bin/bash

SAVE_DIR='./outputs_test/DF_inshop_pred/' 
DATA_ROOT='./dataset/DeepFashion-In-Shop'
DATA_LIST_PATH='dataset/list/DeepFashion-In-Shop_list.txt'
BS=16
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots/LIP_epoch_149.pth'
NUM_CLASSES=20

python3 infer_mirror.py --save-dir ${SAVE_DIR} \
                        --data-dir ${DATA_ROOT} \
                        --list-path ${DATA_LIST_PATH} \
                        --batch-size ${BS} \
                        --num-classes ${NUM_CLASSES}\
                        --restore-from ${SNAPSHOT_FROM}\
                        --gpu ${GPU_IDS} \
                        --input-size ${INPUT_SIZE}\
                        --mirror
