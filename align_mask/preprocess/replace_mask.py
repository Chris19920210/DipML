# coding=utf-8
import nltk
import random
import jieba
import jieba.posseg as pseg

def get_alignment_pairs(lalign):
    str_pairs = lalign.split(' ')
    ez_pairs, ze_pairs = {}, {}
    for sp in str_pairs:
        if '-' in sp:
            cols = sp.split('-')
            zid, eid = int(cols[0]), int(cols[1])
            if eid not in ez_pairs:
                ez_pairs[eid] = [zid]
            else:
                ez_pairs[eid].append(zid)
            if zid not in ze_pairs:
                ze_pairs[zid] = [eid]
            else:
                ze_pairs[zid].append(eid)
    return ez_pairs, ze_pairs

#nltk.download('averaged_perceptron_tagger')  # run for the first time
jieba.load_userdict('../corpus/dicts.txt')
num_masks = 50
masks = []
for i in range(num_masks):
    if i < 9:
        masks.append('DIPML00' + str(i + 1) + 'MASK')
    else:
        masks.append('DIPML0' + str(i + 1) + 'MASK')
print('masks:', masks)

# pos tags of nouns for nltk-english and jieba-chinese
nltk_tags = ['NN', 'NNS', 'NNP', 'NNPS']
jieba_tags = ['n', 'nr', 'ns', 'nt', 'nz']

def is_continuous(arr):
    is_cont = True
    arr.sort()
    len_arr = len(arr)
    if len_arr == 0:
        return False
    for i in range(len_arr - 1):
        if arr[i] + 1 != arr[i + 1]:
            is_cont = False
            break
    return is_cont

# return: whether to mask; sum to sid; start of src mask; len of src mask; start of tgt mask; len of tgt mask
def check_noun_phrase(sid, src_tagged, tgt_tagged, ez_aligns, ze_aligns, src_tagset, tgt_tagset):
    # 1. check n
    stag = src_tagged[sid]
    if stag[1] not in src_tagset:
        return False, 1, 0, 0, 0, 0
    if sid not in ez_aligns:
        return False, 1, 0, 0, 0, 0
    tids = ez_aligns[sid]
    len_tids = len(tids)
    # 2. check phrases: many to many
    if not is_continuous(tids):
        return False, 1, 0, 0, 0, 0
    any_noun = False
    for tid in tids:
        if tgt_tagged[tid][1] in tgt_tagset:
            any_noun = True
    if not any_noun:
        return False, 1, 0, 0, 0, 0
    rev_sids = ze_aligns[tids[0]]
    for i in range(1, len_tids):
        tmp_sids = ze_aligns[tids[i]]
        rev_sids = list(set(rev_sids).intersection(set(tmp_sids)))
    if not is_continuous(rev_sids):
        return False, 1, 0, 0, 0, 0
    if random.randint(1, 200) <= 6:
        tids.sort()
        rev_sids.sort()
        return True, rev_sids[-1] - sid + 1, rev_sids[0], len(rev_sids), tids[0], len_tids # many to many hit
    return False, 1, 0, 0, 0, 0

# replace at most #max_masks masks
def replace_each_pair(src_tagged, tgt_tagged, ez_aligns, ze_aligns, src_tagset, tgt_tagset, max_masks=3):
    len_src = len(src_tagged)
    len_tgt = len(tgt_tagged)
    if len_src < 10 or len_tgt < 10:
        max_masks = 1
    elif len_src < 20 or len_tgt < 20:
        max_masks = 2
    sid, masked_cnt = 0, 0
    masked = False
    swords, twords = [-2] * len_src, [-2] * len(tgt_tagged) # record: -2 for keep, -1 for ignore, 0-49 for mask
    while sid < len_src:
        #print(sid, sword, stag)
        is_mask, sid_append, src_start, src_app, tgt_start, tgt_app = check_noun_phrase(sid, src_tagged, tgt_tagged, ez_aligns, ze_aligns, src_tagset, tgt_tagset)
        #print('==>', is_mask, sid_append, src_start, src_app, tgt_start, tgt_app)
        if masked_cnt >= max_masks:
            break
        if is_mask:
            masked = True
            mid = random.randint(0, num_masks - 1)
            swords[src_start] = mid
            for k in range(src_start + 1, min(len_src, src_start + src_app)):
                swords[k] = -1
            twords[tgt_start] = mid
            for k in range(tgt_start + 1, min(len_tgt, tgt_start + tgt_app)):
                twords[k] = -1
            sid += sid_append + 2 # the mask cannot be too close
        else:
            sid += 1
    #print('sword', swords)
    #print('tword', twords)
    src_sent, tgt_sent = [], []
    for stag, sw in zip(src_tagged, swords):
        if sw == -2:
            src_sent.append(stag[0])
        elif sw >= 0:
            src_sent.append(masks[sw])
    for ttag, tw in zip(tgt_tagged, twords):
        if tw == -2:
            tgt_sent.append(ttag[0])
        elif tw >= 0:
            tgt_sent.append(masks[tw])
    return masked, ' '.join(src_sent), ' '.join(tgt_sent)


# get pos tags
corpus_dir = '../corpus/new'
fzh = open('/home/wudong/align_mask/corpus/nmt_datasets/wmt_zhen_32000k_tok_train.lang1', 'r')
fen = open('/home/wudong/align_mask/corpus/nmt_datasets/wmt_zhen_32000k_tok_train.lang2', 'r')
falign_enzh = open(corpus_dir + '/align.zh-en', 'r')
#falign_zhen = open(corpus_dir + '/align.zh-en', 'r')
mez_en = open(corpus_dir + '/masked_enzh_train.en', 'w')
mez_zh = open(corpus_dir + '/masked_enzh_train.zh', 'w')
mze_en = open(corpus_dir + '/masked_zhen_train.en', 'w')
mze_zh = open(corpus_dir + '/masked_zhen_train.zh', 'w')
cnt = 0
enl, zhl, ezl = fen.readline().replace('\n', ''), fzh.readline().replace('\n', ''), falign_enzh.readline().replace('\n', '')
while enl is not None and enl != '' and zhl is not None and zhl != '' and  ezl is not None and ezl != '':
    #print(enl)
    #print(zhl)
    #print(ezl)
    en_tokens = enl.split(' ')
    #zh_tokens = zhl.split(' ')
    en_tagged = nltk.pos_tag(en_tokens)
    zh_pseg_tagged = pseg.cut(zhl)
    #print(en_tagged)
    #print(zh_tagged)
    zh_tagged = []
    for ztag in zh_pseg_tagged:
        if ztag.word != '' and ztag.word != ' ':
            #print(ztag.word, ztag.flag)
            zh_tagged.append([ztag.word, ztag.flag])
    ez_pairs, ze_pairs = get_alignment_pairs(ezl)
    #ze_pairs = get_alignment_pairs(zel)
    #print(ez_pairs)
    #print(ze_pairs)
    #print(replace_each_pair(en_tagged, zh_tagged, ez_pairs, ze_pairs, nltk_tags, jieba_tags))
    #print("======")
    #print(replace_each_pair(zh_tagged, en_tagged, ze_pairs, ez_pairs, jieba_tags, nltk_tags))
    #break
    ez_masked, nezen, nezzh = replace_each_pair(en_tagged, zh_tagged, ez_pairs, ze_pairs, nltk_tags, jieba_tags)
    if ez_masked:
        mez_en.write(nezen + '\n')
        mez_zh.write(nezzh + '\n')
    ze_masked, nzezh, nzeen = replace_each_pair(zh_tagged, en_tagged, ze_pairs, ez_pairs, jieba_tags, nltk_tags)
    if ze_masked:
        mze_en.write(nzeen + '\n')
        mze_zh.write(nzezh + '\n')
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)
    enl, zhl, ezl = fen.readline().replace('\n', ''), fzh.readline().replace('\n', ''), falign_enzh.readline().replace('\n', '')
fen.close()
fzh.close()
falign_enzh.close()
#falign_zhen.close()
mez_en.close()
mez_zh.close()
mze_en.close()
mze_zh.close()
