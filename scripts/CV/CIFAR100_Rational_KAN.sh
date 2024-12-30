#!/bin/bash


batch_size=512
model=Rational_KAN


for h_dim in 32 64 128 256 512 1024; do
        group=$((h_dim / 4))
        # CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name CIFAR100 \
        #                                 --task_name CV \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group \
        #                                 --need_relu


        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name CIFAR100 \
                                        --task_name CV \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 5 \
                                        --groups $group \
                                        --need_relu

        # CUDA_VISIBLE_DEVICES=0 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name CIFAR100 \
        #                                 --task_name CV \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group \
        #                                 --need_relu
done
