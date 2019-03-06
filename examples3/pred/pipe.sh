#!/usr/bin/env bash

#
CUR_LANG=en     # or other languages
TOOLS_DIR="./"
SRC_DIR="./src/"
MODLE_DIR="./"

bash ${TOOLS_DIR}/udpipe/models/udpipe_tok.sh ${CUR_LANG} | \
PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/examples3/pred/pipe_tp.py --tagger_path ${MODLE_DIR}/zft_${CUR_LANG}/models/ --tagger_name network.pt --parser_path ${MODLE_DIR}/zfp_${CUR_LANG}/models/ --parser_name network.pt --len_thresh_min 0 --len_thresh_max 10000 --oov_thresh 1. --parser_mst 1 --parser_topk 1 --tagger_topk 1
