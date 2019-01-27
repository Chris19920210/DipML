import argparse
from collections import defaultdict
import os
import string
import re
import html
from zhon.hanzi import punctuation as zhpunctuation
from mosestokenizer import MosesTokenizer
import MeCab

parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to base-dir')
parser.add_argument('--input', type=str, default=None,
                    help='path to input')
parser.add_argument('--output', type=str, default=None,
                    help='path to output')
args = parser.parse_args()



class EnZhTuple(object):
    def __init__(self, zh, en):
        self.zh = zh
        self.en = en

    def setZh(self, zh):
        self.zh = zh

    def setEn(self, en):
        self.en = en

    def getZh(self):
        return self.zh

    def getEn(self):
        return self.en


def punc_remover_lower(en_sentence):
    global table
    return en_sentence.translate(table).lower()


def check_and_add(en_sentence, zh_sentence):
    global sentence_dict, entokenzier, mecab
    key = punc_remover_lower(en_sentence)
    if key not in sentence_dict:
        sentence_dict[key] = EnZhTuple(zh_sentence, en_sentence)
    else:
        if abs(len(entokenzier(sentence_dict[key].getEn())) - len(mecab.parse(zh_sentence))) \
                < abs(len(entokenzier(sentence_dict[key].getEn())) - len(mecab.parse(sentence_dict[key].getZh()))):
            sentence_dict[key].setZh(zh_sentence)


if __name__ == '__main__':
    sentence_dict = defaultdict(EnZhTuple)
    punctuations = string.punctuation + zhpunctuation
    punctuations = punctuations.replace("(", "").replace("\"", "").replace("<", "") + " \t"
    cleaner = re.compile('<.*?>')
    table = str.maketrans(dict.fromkeys(string.punctuation + zhpunctuation + "\t "))
    entokenzier = MosesTokenizer('en')
    mecab = MeCab.Tagger('-Owakati')

    with open(os.path.join(args.base_dir, args.input), 'r', errors='ignore') as f:
        for each in f.readlines():
            try:
                each = bytes(each, 'utf-8').decode('utf-8', 'ignore')
                each = each.strip().replace(u'\uf020', "")
                each = each.replace('\n', '')
            except Exception as e:
                print(str(e))
            try:
                # if each[0].isdigit() or each[0] in punctuations:
                #     continue
                #cleantext = re.sub(cleaner, '', each)
                cleantext = html.unescape(each)
                line = cleantext.strip().split("\t")
                line = list(filter(lambda x: x != "", line))
                check_and_add(line[0].strip(), line[1].strip())
            except Exception as e:
                print("""
                Sentence:{sentence:s}
                Error:{error:s}
                """.format(sentence=each, error=str(e)))

    with open(os.path.join(args.base_dir, args.output), 'w') as g:
        for _, value in sentence_dict.items():
            g.write(value.getEn())
            g.write('\t')
            g.write(value.getZh())
            g.write('\n')

