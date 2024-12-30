#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data
model=RBF_KAN


for h_dim in 256 512 1024 ; do
        group=$((h_dim / 4))


        CUDA_VISIBLE_DEVICES=2 python ../main_nlp.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name AG_NEWS \
                                        --task_name NLP \
                                        --grid_size 20 \
                                        --lr 0.001 \
                                        --grid_range -10 10 \
                                        --hidden_layer $h_dim \
                                        --spline_order 3 \
                                        --need_relu 


        # CUDA_VISIBLE_DEVICES=2 python ../main_nlp.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name AG_NEWS \
        #                                 --task_name NLP \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim $h_dim \
        #                                 --spline_order 3 \
        #                                 --need_relu 


        # CUDA_VISIBLE_DEVICES=2 python ../main_nlp.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name AG_NEWS \
        #                                 --task_name NLP \
        #                                 --grid_size 20 \
        #                                 --lr 0.001 \
        #                                 --grid_range -10 10 \
        #                                 --hidden_layer $h_dim \
        #                                 --spline_order 3 \
        #                                 --need_relu 
done
