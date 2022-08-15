#!/bin/bash
DATA_DIR='../datasets/final/iemocap'
MODEL_NAME_OR_PATH='../roberta-base'
OUTPUT_DIR='../output_i_csf'
POSITION='relative'
SPECIFIC='iemocap'
KNOWLEDGE='CSF'

CUDA_VISIBLE_DEVICES='0' python ../run.py \
--data_dir $DATA_DIR \
--seed 44 \
--do_train \
--knowledge_mode $KNOWLEDGE \
--evaluate_after_epoch \
--model_name_or_path $MODEL_NAME_OR_PATH \
--output_dir $OUTPUT_DIR \
--max_seq_length 128 \
--logging_steps -1 \
--per_gpu_train_batch_size 1 \
--gradient_accumulation_steps 4 \
--per_gpu_eval_batch_size 32 \
--learning_rate 3e-5 \
--dropout_prob 0.1 \
--num_train_epochs 20 \
--position_mode $POSITION \
--specific $SPECIFIC \
--window_size 5 \
--n_position 120 \
--d_inner 3072
