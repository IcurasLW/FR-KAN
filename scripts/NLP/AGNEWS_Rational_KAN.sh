#!/bin/bash


batch_size=512
data_path=/home/nathan/KAN_nathan/efficient-kan/data
model=Rational_KAN


for h_dim in 1024; do
        group=$((h_dim / 4))
        CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name AG_NEWS \
                                        --task_name NLP \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim \
                                        --spline_order 5 \
                                        --groups $group \
                                        --need_relu

        CUDA_VISIBLE_DEVICES=1 python ../main_nlp.py \
                                        --model $model \
                                        --batch_size $batch_size \
                                        --data_path ../data \
                                        --data_name AG_NEWS \
                                        --task_name NLP \
                                        --lr 0.002 \
                                        --hidden_layer $h_dim $h_dim \
                                        --spline_order 5 \
                                        --groups $group \
                                        --need_relu


        # CUDA_VISIBLE_DEVICES=2 python ../main_nlp.py \
        #                                 --model $model \
        #                                 --batch_size $batch_size \
        #                                 --data_path ../data \
        #                                 --data_name AG_NEWS \
        #                                 --task_name NLP \
        #                                 --lr 0.002 \
        #                                 --hidden_layer $h_dim $h_dim $h_dim \
        #                                 --spline_order 5 \
        #                                 --groups $group \
        #                                 --need_relu
done
