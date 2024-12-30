#!/bin/bash


batch_size=64
model=KAN


for h_dim in 128 ; do
        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim\
                                        --spline_order 3 \


        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 


        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim $h_dim\
                                        --spline_order 3 
done