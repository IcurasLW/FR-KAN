#!/bin/bash


batch_size=512
model=KAN


# 32 64 set smooth parameter to 0.03
# other smooth param to 0.1

for h_dim in 128; do
        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name CIFAR10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -1 1 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 

        # CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name CIFAR10 \
        #                                 --task_name CV \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --spline_order 3 

        # CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name CIFAR10 \
        #                                 --task_name CV \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 
done
