#!/bin/bash


batch_size=128
data_path=/home/nathan/KAN_nathan/efficient-kan/data

for h_dim in 128; do
        # CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
        #                                 --model KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AG_NEWS \
        #                                 --grid_size 30 \
        #                                 --task_name NLP \
        #                                 --grid_range -15 15 \
        #                                 --hidden_layer $h_dim \
        #                                 --lr 0.002 \
        #                                 --spline_order 3 
                                        # --need_relu

        CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
                                        --model KAN \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AG_NEWS \
                                        --grid_size 10 \
                                        --task_name NLP \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim \
                                        --lr 0.002 \
                                        --spline_order 3 \
                                        --need_relu

        # CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
        #                                 --model KAN \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AG_NEWS \
        #                                 --grid_size 20 \
        #                                 --task_name NLP \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --lr 0.002 \
        #                                 --spline_order 3
done