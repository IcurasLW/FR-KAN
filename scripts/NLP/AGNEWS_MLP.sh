#!/bin/bash

batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data
for h_dim in 64; do
        # CUDA_VISIBLE_DEVICES=0 python ../main_nlp.py \
        #                                 --model MLP \
        #                                 --batch_size $batch_size \
        #                                 --data_path $data_path \
        #                                 --data_name AG_NEWS \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim 

        CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AG_NEWS \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim 

        CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path $data_path \
                                        --data_name AG_NEWS \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim $h_dim 
done