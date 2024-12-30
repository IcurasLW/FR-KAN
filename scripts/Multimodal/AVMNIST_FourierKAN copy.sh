#!/bin/bash


batch_size=512
model=Fourier_KAN

for h_dim in 4 8 16 32 64 128; do
        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name AVMNIST \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim 




        CUDA_VISIBLE_DEVICES=1 python ../main_mm.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name AVMNIST \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --hidden_layer $h_dim $h_dim 
done