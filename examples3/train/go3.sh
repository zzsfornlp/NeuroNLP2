#!/usr/bin/env bash

#
CUR_DIR=`pwd`
#for cur_lang in es it ja; do
#for cur_lang in en "fi" fr; do
#for cur_lang in ru de; do
#for cur_lang in cs zhs; do
for cur_lang in "fi" en de fr es it cs ru ja zhs; do
# freeze-tagger
cd ${CUR_DIR}
mkdir -p zt_${cur_lang}_freeze
cd zt_${cur_lang}_freeze
RGPU=1 CUR_LANG=${cur_lang} bash ../train_tagger_freeze.sh |& tee log_train
# tagger
cd ${CUR_DIR}
mkdir -p zt_${cur_lang}
cd zt_${cur_lang}
RGPU=1 CUR_LANG=${cur_lang} bash ../train_tagger.sh |& tee log_train
# freeze-parser
cd ${CUR_DIR}
mkdir -p zp_${cur_lang}_freeze
cd zp_${cur_lang}_freeze
RGPU=1 CUR_LANG=${cur_lang} bash ../train_parser_freeze.sh |& tee log_train
# parser
cd ${CUR_DIR}
mkdir -p zp_${cur_lang}
cd zp_${cur_lang}
RGPU=1 CUR_LANG=${cur_lang} bash ../train_parser.sh |& tee log_train
done

#####
for cur_lang in en zhs ja "fi" de fr it es ru cs; do
    # nopos + freeze
    cd ${CUR_DIR}
    mkdir -p zp_${cur_lang}_nopos_freeze
    cd zp_${cur_lang}_nopos_freeze
    CUR_LANG=${cur_lang} bash ../train_parser_nopos_freeze.sh |& tee log_train
done

