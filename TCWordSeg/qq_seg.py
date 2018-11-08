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
    num = TCGetResulqtCnt(seg_handle)
    words = []
    pos_list = []
    for i in range(num):
        wordpos = TCGetAt(seg_handle, i);
        word = wordpos.word
        pos = wordpos.pos
        words.append(word)
        pos_list.append(str(pos))
        #seg += '%s ' %(word,)

    return words#,pos_list

if __name__ == '__main__':
    seghandle = GetSegHandle('dict/')
    print "Init Complete"
    cnt = 0
    for line in sys.stdin:
        if cnt % 10000 == 0:
            sys.stderr.write('%d lines \n' % cnt)
        cnt += 1
        #vec = line.strip().split('\t')
        #if len(vec) != 3:
        #    continue
        #words_title = GetSegWords(vec[2] ,seghandle)
        #print vec[0] + '\t' + vec[1] + '\t' +  ' '.join(words_title)
        words_title = GetSegWords(line ,seghandle)
	print ' '.join(words_title)
        '''
        words_content = GetSegWords(vec[3],seghandle)
        #words_summary = GetSegWords(vec[1],seghandle)
        print vec[0] + '\t' + vec[2] + '\t' + vec[4] + '\t' +  ' '.join(words_title)  + '\t' + ' '.join(words_content)#+ '\t' + ' '.join(pos_list)#+ '\t' + ' '.join(words_summary)
        '''
