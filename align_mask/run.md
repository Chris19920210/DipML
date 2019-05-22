# 0. 安装fast_align, https://github.com/clab/fast_align
# 1. 选取已分词的中英文文件路径，修改 generate_fast_align_corpurs.py 路径并运行
# 2. 对齐
```
/home/wudong/align_mask/fast_align/build/fast_align -i ../corpus/new/train.en-zh -d -o -v > ../corpus/new/forward.align
/home/wudong/align_mask/fast_align/build/fast_align -i ../corpus/new/train.en-zh -d -o -v -r > ../corpus/new/reverse.align
/home/wudong/align_mask/fast_align/build/atools -i ../corpus/new/forward.align -j ../corpus/new/reverse.align -c grow-diag-final-and > ../corpus/new/align.en-zh
/home/wudong/align_mask/fast_align/build/atools -j ../corpus/new/forward.align -i ../corpus/new/reverse.align -c grow-diag-final-and > ../corpus/new/align.zh-en
```
