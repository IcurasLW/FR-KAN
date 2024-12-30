#!/bin/bash


batch_size=512
model=Knot_KAN

############################### One-Layer Network ###############################
# 32 64 set smooth parameter to 0.01
# other smooth param to 0.05
# for h_dim in 32 64 ; do
for h_dim in 32 64 128 256 512 1024; do
        group=16
        # CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name MNIST \
        #                                 --task_name CV \
        #                                 --grid_size 30 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.05 \
        #                                 --groups $group \
        #                                 --smooth \
        #                                 --need_relu 


        CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name MNIST \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.03 \
                                        --groups $group \
                                        --smooth \
                                        --need_relu 


        CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name MNIST \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.05 \
                                        --groups $group \
                                        --smooth \
                                        --need_relu 
done
