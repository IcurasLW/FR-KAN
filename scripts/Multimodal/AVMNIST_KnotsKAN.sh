#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data









for h_dim in 256 512 1024; do
        group=16
        # CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AVMNIST \
        #                                 --grid_size 20 \
        #                                 --task_name MM \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --lr 0.002 \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.005 \
        #                                 --groups $group \
        #                                 --need_relu 
                                        # --smooth 


        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model Knot_KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AVMNIST \
                                        --grid_size 20 \
                                        --grid_range -10 10 \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.005 \
                                        --groups $group \
                                        --need_relu \
                                        --smooth 



        # CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AVMNIST \
        #                                 --grid_size 20 \
        #                                 --grid_range -10 10 \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.005 \
        #                                 --groups $group \
        #                                 --need_relu \
        #                                 --smooth 
done
