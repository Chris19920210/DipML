#!/bin/bash

WORKSPACE=`pwd`
mosehome=$WORKSPACE/mosesdecoder
DATA_DIR=$WORKSPACE/data
CORPUS=$WORKSPACE/tmp/raw_english

cat $CORPUS | $mosehome/scripts/tokenizer/tokenizer.perl -l en > $WORKSPACE/tmp/en.total.tmp

$mosehome/scripts/recaser/train-truecaser.perl --model $WORKSPACE/MODEL --corpus $WORKSPACE/tmp/en.total.tmp

rm -r $WORKSPACE/tmp/en.total.tmp
