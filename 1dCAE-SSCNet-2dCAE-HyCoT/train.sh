#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DEVICES=0
NUM_WORKERS=4

MODEL=hycot_cr4

LEARNING_RATE=1e-3
EPOCHS=2_000

MODE=easy
TRANSFORM=random_16x16

TRAIN_BATCH_SIZE=2
VAL_BATCH_SIZE=2

SAVE_DIR=./results/trains/

LOSS=mse

nohup \
  python -u train.py \
    --devices ${DEVICES} \
    --train-batch-size ${TRAIN_BATCH_SIZE} \
    --val-batch-size ${VAL_BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --learning-rate ${LEARNING_RATE} \
    --mode ${MODE} \
    --model ${MODEL} \
    --loss ${LOSS} \
    --transform ${TRANSFORM} \
    --epochs ${EPOCHS} \
    --save-dir ${SAVE_DIR} \
  &> results/logs/${DEVICES}.log &
exit
