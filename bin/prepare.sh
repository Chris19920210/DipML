#!/bin/sh

#!/bin/sh
WORKSPACE=`pwd`
DATA_DIR=$WORKSPACE/data
mosehome=$WORKSPACE/mosesdecoder
DICT=$WORKSPACE/vocabulary/dicts.txt

#/home/chenrihan/mosesdecoder/scripts/tokenizer
#chinese words segmentation
cat $DATA_DIR/zh_train/* | python $WORKSPACE/jieba_cws.py --dict $DICT| $mosehome/scripts/recaser/detruecase.perl > $DATA_DIR/zh.total.1 &
#cat $DATA_DIR/zh_test/* | python prepare_data/jieba_cws.py | $mosehome/scripts/recaser/detruecase.perl > $DATA_DIR/ai_challenger_en-zh.zh.tok.test &

## Tokenize the english word and truecase it
cat $DATA_DIR/en_train/* | $mosehome/scripts/tokenizer/tokenizer.perl -l en | $mosehome/scripts/recaser/truecase.perl --model $WORKSPACE/MODEL | python tag_remover.py |$mosehome/scripts/recaser/detruecase.perl > $DATA_DIR/en.total.1 &
#cat $DATA_DIR/en_test/* | $mosehome/scripts/tokenizer/tokenizer.perl -l en  | $mosehome/scripts/recaser/truecase.perl --model MODEL | $mosehome/scripts/recaser/detruecase.perl > $DATA_DIR/ai_challenger_en-zh.en.tok.test &
