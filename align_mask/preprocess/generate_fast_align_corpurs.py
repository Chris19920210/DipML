corpus_dir = '../corpus/new'
fen = open('/home/wudong/align_mask/corpus/nmt_datasets/wmt_zhen_32000k_tok_train.lang1', 'r')
fzh = open('/home/wudong/align_mask/corpus/nmt_datasets/wmt_zhen_32000k_tok_train.lang2', 'r')
fcorp = open(corpus_dir + '/train.zh-en', 'w')
cnt = 0
enl, zhl = fen.readline().replace('\n', ''), fzh.readline().replace('\n', '')
while enl is not None and enl != '' and zhl is not None and zhl != '':
    fcorp.write(enl + ' ||| ' + zhl + '\n')
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
    enl, zhl = fen.readline().replace('\n', ''), fzh.readline().replace('\n', '')
fen.close()
fzh.close()
fcorp.close()
