#!/bin/bash


batch_size=128
model=Knot_KAN


for h_dim in 32 64 128 256 512 1024; do
        group=$((h_dim / 4))
        # CUDA_VISIBLE_DEVICES=3 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name STL10 \
        #                                 --task_name CV \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0.001 \
        #                                 --groups $group \
        #                                 --smooth 



        CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --lr 0.002 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim $h_dim\
                                        --spline_order 3 \
                                        --smooth_lambda 0 \
                                        --groups $group \
                                        --smooth \
                                        --need_relu



        # CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name STL10 \
        #                                 --task_name CV \
        #                                 --grid_size 30 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --smooth_lambda 0 \
        #                                 --groups $group \
        #                                 --smooth \
        #                                 --need_relu
done



# for h_dim in 128 256 512 1024; do
#         group=$((h_dim / 4))
#         CUDA_VISIBLE_DEVICES=2 python ../main_cv.py \
#                                         --model $model \
#                                         --batch_size $batch_size \
#                                         --data_path ../data \
#                                         --data_name STL10 \
#                                         --task_name CV \
#                                         --grid_size 20 \
#                                         --lr 0.001 \
#                                         --grid_range -10 10 \
#                                         --hidden_layer $h_dim $h_dim\
#                                         --spline_order 3 \
#                                         --smooth_lambda 0.0001 \
#                                         --groups $group \
#                                         --smooth 
# done