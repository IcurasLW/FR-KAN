#!/bin/bash


batch_size=128
model=Fourier_KAN

for h_dim in 4 8 16 32 64 128; do
        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name CIFAR10 \
                                        --task_name CV \
                                        --grid_size 10 \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim 


        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name CIFAR10 \
                                        --task_name CV \
                                        --grid_size 10 \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim $h_dim 
done