import MeCab
import sys
import argparse

parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--dict', type=str, default=None,
                    help='path to dictionary')
args = parser.parse_args()


def mecab_cws(string):
    global mecab
    seg = mecab.parse(string.strip())
    return seg.strip()


if __name__ == '__main__':
    if args.dict:
        mecab = MeCab.Tagger('-Owakati -d {0}'.format(args.dict))
    else:
        mecab = MeCab.Tagger('-Owakati')

    for line in sys.stdin:
        line_cws = mecab_cws(line)
        sys.stdout.write(line_cws)
        sys.stdout.write('\n')
