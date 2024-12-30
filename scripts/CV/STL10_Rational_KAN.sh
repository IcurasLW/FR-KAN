#!/bin/bash


batch_size=512
model=Rational_KAN


for h_dim in 32 64 128 256 512 1024; do
        group=$((h_dim / 4))
        CUDA_VISIBLE_DEVICES=3 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim \
                                        --spline_order 5 \
                                        --groups $group 



        CUDA_VISIBLE_DEVICES=3 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --groups $group 
                                        # --need_relu



        # CUDA_VISIBLE_DEVICES=3 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name STL10 \
        #                                 --task_name CV \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group 
done
