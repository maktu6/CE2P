#!/bin/bash

SAVE_DIR='/home/maktub/Downloads/public_dataset/DeepFashion/fashionGAN_dataset/CE2P_parse_for_img'
DATA_ROOT='/home/maktub/Downloads/public_dataset/DeepFashion/fashionGAN_dataset/img'
BS=16
NUM_CPU=2
GPU_IDS='0'
INPUT_SIZE='384,384'
SNAPSHOT_FROM='./snapshots/LIP_epoch_149.pth'
NUM_CLASSES=20


python3 infer_mirror.py --save-dir ${SAVE_DIR} \
                        --data-dir ${DATA_ROOT} \
                        --batch-size ${BS} \
                        --num_workers ${NUM_CPU} \
                        --num-classes ${NUM_CLASSES}\
                        --restore-from ${SNAPSHOT_FROM}\
                        --gpu ${GPU_IDS} \
                        --input-size ${INPUT_SIZE}\
                        --mirror