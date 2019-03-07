DATA_DIRECTORY='./dataset/mvc_images'
DATA_LIST_PATH='./dataset/list/mvc_list.txt'
NUM_CLASSES=20 
#RESTORE_FROM='./snapshots/LIP_CE2P_trainVal_321_681.pth'
RESTORE_FROM='./snapshots/LIP_CE2P_trainVal_473.pth'
SAVE_DIR='./outputs_test/mvc_pred/' 
#INPUT_SIZE='321,681'
INPUT_SIZE='473,473'
GPU_ID=0
 
python3.6 test.py --data-dir ${DATA_DIRECTORY} \
                --data-list ${DATA_LIST_PATH} \
                --input-size ${INPUT_SIZE} \
                --is-mirror \
                --num-classes ${NUM_CLASSES} \
                --save-dir ${SAVE_DIR} \
                --gpu ${GPU_ID} \
                --restore-from ${RESTORE_FROM}
