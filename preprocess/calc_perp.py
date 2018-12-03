#coding=utf-8

import math
import re


def load_vcbs(psrc):
    fsrc = open(psrc)
    w2i = {}
    for line in fsrc:
        cols = line.replace('\n', '').split(' ')
        idx = int(cols[0])
        word = cols[1]
        w2i[word] = idx
    fsrc.close()
    return w2i


def load_trans(ptrans):
    ftrans = open(ptrans)
    p4ij = {}
    for line in ftrans:
        cols = line.replace('\n', '').split(' ')
        srcid = int(cols[0])
        trgid = int(cols[1])
        prob = float(cols[2])
        if srcid not in p4ij:
            p4ij[srcid] = {}
        p4ij[srcid][trgid] = prob
    ftrans.close()
    return p4ij


def get_prob(sid, tid, p4ij, default_prob):
    if sid not in p4ij:
        return default_prob
    if tid not in p4ij[sid]:
        return default_prob
    return p4ij[sid][tid]


def parse_alignments(target_line, alignments_line, src_vcbs, trg_vcbs, trans, default_prob=0.754821):
    trgs = target_line.split(' ')
    cur_tids = {}
    for i, trg in enumerate(trgs):
        if trg == '': continue
        cur_tids[i+1] = trg_vcbs[trg]
    word_alignments_regex = "(\S+)\s\(\{([\s\d]*)\}\)"
    alignments = re.findall(word_alignments_regex, alignments_line)
    perplexity = .0
    absol = .0
    for w, idstr in alignments:
        if 'NULL' == w:
            #print('defailt prob')
            perplexity += math.log(default_prob)
            absol += default_prob
        else:
            sid = src_vcbs[w]
            tids = idstr[1:-1].split(' ')
            for t in tids:
                if t == '':
                    continue
                tid = cur_tids[int(t)]
                prob = get_prob(sid, tid, trans, default_prob)
                perplexity += math.log(prob)
                absol += abs(prob)
    perplexity = math.exp(-perplexity / absol)
    return perplexity


def load_aligns(palign, src_vcbs, trg_vcbs, trans):
    falign = open(palign)
    fres = open(direct + '_res.txt', 'w')
    line = falign.readline().replace('\n', '')
    ltrg = falign.readline().replace('\n', '')
    lsrc = falign.readline().replace('\n', '')
    while line is not None and line != '':
        perp = parse_alignments(ltrg, lsrc, src_vcbs, trg_vcbs, trans)
        score = float(line.split(' ')[-1])
        fres.write(ltrg + '\t' + str(perp) + '\t' + str(score) + '\n')
        line = falign.readline().replace('\n', '')
        ltrg = falign.readline().replace('\n', '')
        lsrc = falign.readline().replace('\n', '')
    fres.close()
    falign.close()

import sys
direct = sys.argv[1]
src_vcbs = load_vcbs(direct + '/' + direct + '.trn.src.vcb')
trg_vcbs = load_vcbs(direct + '/' + direct + '.trn.trg.vcb')
trans = load_trans(direct + '/' + direct + '.t3.final')
load_aligns(direct + '/' + direct + '.A3.final', src_vcbs, trg_vcbs, trans)
# src_vcbs = load_vcbs('test/e2z/e2z.trn.src.vcb')
# trg_vcbs = load_vcbs('test/e2z/e2z.trn.trg.vcb')
# trans = load_trans('test/e2z/e2z.t3.final')
# load_aligns('test/e2z/e2z.A3.final', src_vcbs, trg_vcbs, trans)
