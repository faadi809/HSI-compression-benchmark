#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

DEVICE=0
NUM_WORKERS=4

MODE=easy

MODEL=hycot_cr4
BATCH_SIZE=1

CHECKPOINT=./results/weights/hycot_cr4.pth.tar

nohup \
  python -u test.py \
    --device ${DEVICE} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --mode ${MODE} \
    --model ${MODEL} \
    --checkpoint ${CHECKPOINT} \
  > results/logs/${DEVICE}.log | tail -F logs/${DEVICE}.log
