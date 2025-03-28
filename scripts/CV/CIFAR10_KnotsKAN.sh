#!/bin/bash


batch_size=512
model=Knot_KAN


# 32 64 set smooth parameter to 0.03
# other smooth param to 0.1

# for h_dim in 32 64; do
#         group=$((h_dim / 2))
#         CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
#                                         --model $model \
#                                         --batch_size $batch_size \
#                                         --data_path ../data \
#                                         --data_name CIFAR10 \
#                                         --task_name CV \
#                                         --grid_size 20 \
#                                         --lr 0.002 \
#                                         --grid_range -10 10 \
#                                         --hidden_layer $h_dim $h_dim $h_dim \
#                                         --spline_order 3 \
#                                         --smooth_lambda 0.001 \
#                                         --groups $group \
#                                         --need_relu
#                                         # --smooth \
#                                         # --need_relu 
# done



for h_dim in 128 ; do
        group=64
        # CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name CIFAR10 \
        #                                 --task_name CV \
        #                                 --grid_size 50 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.001 \
        #                                 --groups $group \
        #                                 --need_relu 
        #                                 # --smooth 
        #                                 # --need_relu 



        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name CIFAR10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
                                        --smooth \
                                        --need_relu 
done