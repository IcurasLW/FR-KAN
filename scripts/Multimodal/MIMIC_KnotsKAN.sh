#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data



for h_dim in 512 ; do
        group=$((h_dim / 1))
        # group=-1
        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model Knot_KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name MIMIC \
                                        --grid_size 20 \
                                        --grid_range -10 10 \
                                        --lr 0.01 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.002 \
                                        --groups $group \
                                        --need_relu \
                                        --smooth 



        # CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name MIMIC \
        #                                 --grid_size 20 \
        #                                 --grid_range -10 10 \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.005 \
        #                                 --groups $group \
        #                                 --need_relu 
                                        # --smooth 



        # CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name MIMIC \
        #                                 --grid_size 20 \
        #                                 --grid_range -10 10 \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.005 \
        #                                 --groups $group \
        #                                 --need_relu 
        #                                 --smooth 
done
