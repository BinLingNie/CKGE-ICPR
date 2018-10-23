#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--data_dir ../data/FB15k/ \
--skp_file ../data/emd/FB15k/ske_fb15k_400_025_025-embs.npy \
--save_file ../data/result/FB15k/result_TransE_fb15k_400-1000-003-025-025-4096-embs.txt \
--embedding_dim 400 \
--margin_value 1 \
--batch_size 4096 \
--learning_rate 0.003 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 100 \
--max_epoch 1600

CUDA_VISIBLE_DEVICES=0 \
python main.py \
--data_dir ../data/FB15k/ \
--skp_file ../data/emd/FB15k/ske_fb15k_300_025_025-embs.npy \
--save_file ../data/result/FB15k/result_TransE_fb15k_300-1000-003-025-025-4096-embs.txt \
--embedding_dim 300 \
--margin_value 1 \
--batch_size 4096 \
--learning_rate 0.003 \
--n_generator 24 \
--n_rank_calculator 24 \
--eval_freq 100 \
--max_epoch 1600