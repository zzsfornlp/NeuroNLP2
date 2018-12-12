#!/usr/bin/env bash

#
CUR_DIR=`pwd`

for cur_lang in en cs "fi" fr it de zhs ja; do
# tagger
cd ${CUR_DIR}
mkdir -p zt_${cur_lang}
cd zt_${cur_lang}
RGPU=1 CUR_LANG=${cur_lang} bash ../src/examples3/train/train_tagger.sh |& tee log_train
# parser
cd ${CUR_DIR}
mkdir -p zp_${cur_lang}
cd zp_${cur_lang}
RGPU=1 CUR_LANG=${cur_lang} bash ../src/examples3/train/train_parser.sh |& tee log_train
done

# OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 RGPU= CUR_LANG=ja bash ../src/examples3/train/train_parser.sh |& tee log_train