#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data
model=Rational_KAN


for h_dim in 32 64 128 256 512 1024; do
        group=16
        CUDA_VISIBLE_DEVICES=0 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name MIMIC \
                                        --task_name MM \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim \
                                        --spline_order 5 \
                                        --groups $group \
                                        --need_relu


        CUDA_VISIBLE_DEVICES=0 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name MIMIC \
                                        --task_name MM \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 5 \
                                        --groups $group \
                                        --need_relu


        # CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name MIMIC \
        #                                 --task_name MM \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group \
        #                                 --need_relu
done
