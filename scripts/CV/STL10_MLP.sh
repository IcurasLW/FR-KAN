#!/bin/bash


batch_size=128


for h_dim in 32 64 128 256 512 1024; do
        # CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
        #                                 --model MLP \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name STL10 \
        #                                 --task_name CV \
        #                                 --hidden_layer $h_dim 

        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --hidden_layer $h_dim $h_dim 

        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model MLP \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --hidden_layer $h_dim $h_dim $h_dim 
done


