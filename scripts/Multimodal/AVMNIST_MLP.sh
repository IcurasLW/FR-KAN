#!/bin/bash


batch_size=512


for h_dim in 32 64 128 256 512 1024; do
        CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path /home/nathan/KAN_nathan/efficient-kan/data \
                                        --data_name AVMNIST \
                                        --task_name Multimodal \
                                        --hidden_layer $h_dim \
                                        --activation ReLU

        # CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
        #                                 --model MLP \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name AVMNIST \
        #                                 --task_name Multimodal \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --activation ReLU

        # CUDA_VISIBLE_DEVICES=2 python ../main_mm.py \
        #                                 --model MLP \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name AVMNIST \
        #                                 --task_name Multimodal \
        #                                 --hidden_layer $h_dim $h_dim $h_dim\
        #                                 --activation ReLU
done