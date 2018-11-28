# from collections import Counter
import argparse
from string import punctuation
from zhon.hanzi import punctuation as zhpunctuation
# from itertools import chain
import os

parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to data base dir')
parser.add_argument('--input', type=str, required=True,
                    help='input path for ensemble file')
parser.add_argument('--output', type=str, required=True,
                    help='output path for ensemble result')
parser.add_argument('--num', type=int, default=5,
                    help='num for filtering')
args = parser.parse_args()


punctuations = list(punctuation + zhpunctuation)


def punc_remover(sentence):
    return list(filter(lambda x: x not in punctuations, sentence))


# def build_counter(sentences):
#     return Counter(chain(*sentences))


# def threshold_remover(sentence):
#     global counter, num
#     return list(filter(lambda x: counter[x] >= num, sentence))


if __name__ == '__main__':
    with open(os.path.join(args.base_dir, args.input), 'w') as g:
        with open(os.path.join(args.base_dir, args.input), 'r') as f:
            for line in f.readlines():
                g.write(" ".join(punc_remover(line.strip().split(" "))))
                g.write("\n")
