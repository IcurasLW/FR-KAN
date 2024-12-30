#!/bin/bash


batch_size=512
model=KAN


# 32 64 set smooth parameter to 0.03
# other smooth param to 0.1



for h_dim in 8 16 32 64 128 ; do
        # CUDA_VISIBLE_DEVICES=1 python ../main_ts.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name ett_h \
        #                                 --task_name TimeSeries \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 3 \
        #                                 --groups 16 \
        #                                 --seq_len 336 \
        #                                 --pred_len 720 \
        #                                 --features M \
        #                                 --target OT \
        #                                 --max_len -1 \
        #                                 --data_path ETTh1.csv \
        #                                 --percent 100 \
        #                                 --smooth \
        #                                 --need_relu


        CUDA_VISIBLE_DEVICES=1 python ../main_ts.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name ett_h \
                                        --task_name TimeSeries \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim  \
                                        --spline_order 3 \
                                        --groups 16 \
                                        --seq_len 336 \
                                        --pred_len 720 \
                                        --features M \
                                        --target OT \
                                        --max_len -1 \
                                        --data_path ETTh1.csv \
                                        --percent 100 \
                                        --smooth \
                                        --need_relu


        # CUDA_VISIBLE_DEVICES=1 python ../main_ts.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name ett_h \
        #                                 --task_name TimeSeries \
        #                                 --grid_size 20 \
        #                                 --lr 0.01 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --groups 16 \
        #                                 --seq_len 336 \
        #                                 --pred_len 720 \
        #                                 --features M \
        #                                 --target OT \
        #                                 --max_len -1 \
        #                                 --data_path ETTh1.csv \
        #                                 --percent 100 \
        #                                 --smooth \
        #                                 --need_relu

done
