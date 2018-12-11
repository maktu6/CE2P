DATA_DIRECTORY='./dataset/LIP'
DATA_LIST_PATH='./dataset/list/lip/testList.txt' 
NUM_CLASSES=20 
RESTORE_FROM='./snapshots/LIP_CE2P_trainVal_473.pth'
SAVE_DIR='./outputs_test/LIP' 
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
