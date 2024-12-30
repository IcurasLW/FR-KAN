#!/bin/bash


batch_size=128
model=Knot_KAN


# 32 64 set smooth parameter to 0.01
# other smooth param to 0.001
for h_dim in 128 ; do
        if [ $h_dim -gt 256 ]; then
                group=128
        else
                group=$((h_dim / 2))
        fi

        CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name STL10 \
                                        --task_name CV \
                                        --grid_size 20 \
                                        --grid_range -1 1 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --smooth_lambda 0.001 \
                                        --groups $group \
                                        --smooth True 
done



# for h_dim in 64; do
#         if [ $h_dim -gt 256 ]; then
#                 group=64
#         else
#                 # group=$((h_dim / 2))
#                 group=32
#         fi

#         CUDA_VISIBLE_DEVICES=1 python ../main_cv.py \
#                                         --model $model \
#                                         --batch_size $batch_size \
#                                         --data_path ../data \
#                                         --data_name SVHN \
#                                         --task_name CV \
#                                         --grid_size 20 \
#                                         --grid_range -1 1 \
#                                         --hidden_layer $h_dim \
#                                         --spline_order 3 \
#                                         --smooth_lambda 0.01 \
#                                         --groups $group \
#                                         --smooth True 
# done



