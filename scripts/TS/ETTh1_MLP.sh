#!/bin/bash


batch_size=512
model=MLP

for h_dim in 32 64 128 256 512 1024; do
# for h_dim in 1024; do
        # CUDA_VISIBLE_DEVICES=2 python ../main_ts.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name ett_h \
        #                                 --task_name TimeSeries \
        #                                 --lr 0.001 \
        #                                 --hidden_layer $h_dim \
        #                                 --seq_len 336 \
        #                                 --pred_len 720 \
        #                                 --features M \
        #                                 --target OT \
        #                                 --max_len -1 \
        #                                 --data_path ETTh1.csv \
        #                                 --percent 100 


        CUDA_VISIBLE_DEVICES=2 python ../main_ts.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name ett_h \
                                        --task_name TimeSeries \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim $h_dim \
                                        --seq_len 336 \
                                        --pred_len 720 \
                                        --features M \
                                        --target OT \
                                        --max_len -1 \
                                        --data_path ETTh1.csv \
                                        --percent 100 


        CUDA_VISIBLE_DEVICES=2 python ../main_ts.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name ett_h \
                                        --task_name TimeSeries \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim $h_dim $h_dim\
                                        --seq_len 336 \
                                        --pred_len 720 \
                                        --features M \
                                        --target OT \
                                        --max_len -1 \
                                        --data_path ETTh1.csv \
                                        --percent 100 
done