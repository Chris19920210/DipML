#!/bin/bash

WORKSPACE=`pwd`
FILES=$WORKSPACE/data
$WORKSPACE/kenlm/build/bin/lmplz -o 3 --prune 2 2 2 --discount_fallback < $FILES/en.total > $WORKSPACE/en_train.arpa
$WORKSPACE/kenlm/build/bin/lmplz -o 3 --prune 2 2 2 --discount_fallback < $FILES/zh.total > $WORKSPACE/zh_train.arpa
