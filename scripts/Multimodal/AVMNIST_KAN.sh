#!/bin/bash


batch_size=128
data_path=/home/nathan/KAN_nathan/efficient-kan/data

for h_dim in 4 8 16 32 64 128; do
        # CUDA_VISIBLE_DEVICES=0 python ../main_mm.py \
        #                                 --model KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AVMNIST \
        #                                 --grid_size 20 \
        #                                 --task_name MM \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --lr 0.002 \
        #                                 --spline_order 3

        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AVMNIST \
                                        --grid_size 20 \
                                        --task_name MM \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim \
                                        --lr 0.002 \
                                        --spline_order 3

        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AVMNIST \
                                        --grid_size 20 \
                                        --task_name MM \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim $h_dim \
                                        --lr 0.002 \
                                        --spline_order 3
done

