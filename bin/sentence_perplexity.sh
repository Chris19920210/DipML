#!/bin/bash
WORKSPACE=`pwd`
DATA_DIR=$WORKSPACE/data

cat $DATA_DIR/en.total | python sentence_perplexity.py --model-path $WORKSPACE/en_train.arpa > $DATA_DIR/en.total.perplexity
cat $DATA_DIR/zh.total | python sentence_perplexity.py --model-path $WORKSPACE/zh_train.arpa > $DATA_DIR/zh.total.perplexity
