#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data
model=RBF_KAN


for h_dim in 4 8 16 32 64 128; do
        group=16
        CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name MIMIC \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
                                        --need_relu 


        CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name MIMIC \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
                                        --need_relu 


        # CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name MIMIC \
        #                                 --task_name CV \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.001 \
        #                                 --groups $group \
        #                                 --need_relu 

done
