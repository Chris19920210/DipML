#!/usr/bin/python
#coding=gbk
import os, sys, string
from TCWordSeg import *

def GetSegHandle(dict):
    TCInitSeg(dict)
    SEG_MODE = TC_U2L|TC_POS|TC_S2D|TC_U2L|TC_T2S|TC_ENGU|TC_CN
    seg_handle = TCCreateSegHandle(SEG_MODE)
    return seg_handle

def CloseSegHandle(seg_handle):
    TCCloseSegHandle(seg_handle)
    TCUnInitSeg()

def GetSegWords(src, seg_handle):
    TCSegment(seg_handle, src)
    num = TCGetResultCnt(seg_handle)
    words = []
    for i in range(num):
        wordpos = TCGetAt(seg_handle, i);
        word = wordpos.word
        words.append(word)
        #seg += '%s ' %(word,)

    return words

if __name__ == '__main__':
	seghandle = GetSegHandle('dict/')
	#words = GetSegWords("政协委员炮轰“文艺成为市场的奴隶”现象",seghandle)
	for line in sys.stdin:
			vec = line.strip().split('\t')
			terms_1 = GetSegWords(vec[1],seghandle) 
			terms_2 = GetSegWords(vec[2],seghandle) 
			subvec = vec[0].split('###')
			if len(subvec) != 3:
					continue
			print subvec[1] + '###' + subvec[2] + '\t' + ' '.join(terms_1) + '\t' + ' '.join(terms_2)
