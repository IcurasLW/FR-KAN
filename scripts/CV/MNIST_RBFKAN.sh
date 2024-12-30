#!/bin/bash


batch_size=512
model=RBF_KAN


for h_dim in 4 8 16 32 64 128; do
        group=16
        CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
                                       --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name MNIST \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
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
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
                                        --need_relu 


        # CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name MNIST \
        #                                 --task_name CV \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group \
        #                                 --need_relu
done
