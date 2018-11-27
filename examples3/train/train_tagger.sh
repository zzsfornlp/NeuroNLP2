#!/usr/bin/env bash

mkdir -p models tmp

if [[ -z "${SRC_DIR}" ]]; then
SRC_DIR="../src/"
fi

CUDA_VISIBLE_DEVICES=$RGPU PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/examples3/train/tagger.py \
--mode LSTM --num_epochs 200 --batch_size 16 --hidden_size 256 --num_layers 1 \
--char_dim 30 --num_filters 30 --tag_space 256 \
--learning_rate 0.1 --decay_rate 0.05 --schedule 10 --gamma 0.0 \
--dropout std --p_in 0.33 --p_rnn 0.33 0.5 --p_out 0.5 --unk_replace 0.0 \
--model_path "models/" --model_name 'network.pt' \
--word_embedding word2vec --word_path "../data/ud22/zwiki.${CUR_LANG}.vec" \
--train "../data/ud22/${CUR_LANG}_train.conllu" \
--dev "../data/ud22/${CUR_LANG}_dev.conllu" \
--test "../data/ud22/${CUR_LANG}_test.conllu"

# RGPU= CUR_LANG=en bash ../src/examples3/train/train_tagger.sh |& tee log
