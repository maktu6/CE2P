#!/bin/bash

CS_PATH='./dataset/LIP'
BS=16
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots/LIP_epoch_149.pth'
DATASET='val'
NUM_CLASSES=20

python3 evaluate_mirror.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --mirror
