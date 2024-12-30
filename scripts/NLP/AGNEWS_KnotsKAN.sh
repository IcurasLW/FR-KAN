#!/bin/bash


batch_size=128
data_path=/home/nathan/KAN_nathan/efficient-kan/data
for h_dim in 128 ; do
        # group=$((h_dim / 2))
        group=-1

        CUDA_VISIBLE_DEVICES=2 python ../main_nlp.py \
                                        --model Knot_KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AG_NEWS \
                                        --grid_size 10 \
                                        --grid_range -10 10 \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.01 \
                                        --groups $group \
                                        --need_relu
                                        # --smooth \
                                        # --need_relu 
                                        # --smooth 
        
        
        # CUDA_VISIBLE_DEVICES=3 python ../main_nlp.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AG_NEWS \
        #                                 --grid_size 30 \
        #                                 --grid_range -15 15 \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.003 \
        #                                 --groups $group \
        #                                 --smooth
                                        # --need_relu \
                                        # --smooth 


        # CUDA_VISIBLE_DEVICES=3 python ../main_nlp.py \
        #                                 --model Knot_KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AG_NEWS \
        #                                 --grid_size 20 \
        #                                 --grid_range -15 15 \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.001 \
        #                                 --groups $group \
        #                                 --smooth
                                        # --need_relu \
                                        # --smooth 
done
