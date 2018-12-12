#!/usr/bin/env bash

mkdir -p models tmp

if [[ -z "${SRC_DIR}" ]]; then
SRC_DIR="../src/"
fi

CUDA_VISIBLE_DEVICES=$RGPU PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/examples3/train/parser.py \
--mode FastLSTM --num_epochs 500 --batch_size 32 --hidden_size 512 --num_layers 3 \
--pos_dim 50 --char_dim 30 --num_filters 30 --arc_space 512 --type_space 128 \
--opt adam --learning_rate 0.001 --decay_rate 0.75 --epsilon 1e-4 --schedule 10 --gamma 0.0 --clip 5.0 \
--p_in 0.33 --p_rnn 0.33 0.33 --p_out 0.33 --unk_replace 0.5 --pos --char \
--objective cross_entropy --decode mst \
--char_embedding random \
--punctuation '.' '``' "''" ':' ',' 'PUNCT' 'SYM' \
--model_path "models/" --model_name 'network.pt' \
--word_embedding word2vec --word_path "../data/ud23/zwiki.${CUR_LANG}.vec" \
--train "../data/ud23/${CUR_LANG}_train.conllu" \
--dev "../data/ud23/${CUR_LANG}_dev.conllu" \
--test "../data/ud23/${CUR_LANG}_test.conllu"

# RGPU= CUR_LANG=en bash ../src/examples3/train/train_parser.sh |& tee log
