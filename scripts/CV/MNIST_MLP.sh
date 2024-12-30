#!/bin/bash


batch_size=512

for h_dim in 32 64 128 256 512 1024; do
# for h_dim in 128; do
        # CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
        #                                 --model MLP \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name MNIST \
        #                                 --task_name CV \
        #                                 --hidden_layer $h_dim \


        CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name MNIST \
                                        --task_name CV \
                                        --hidden_layer $h_dim $h_dim \
                                        --activation ReLU

        CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name MNIST \
                                        --task_name CV \
                                        --hidden_layer $h_dim $h_dim $h_dim \
                                        --activation ReLU
done