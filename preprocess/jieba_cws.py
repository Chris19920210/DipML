import sys
import jieba


def jieba_cws(string):
    seg_list = jieba.cut(string.strip())
    return ' '.join(seg_list)


if __name__ == '__main__':
    for line in sys.stdin:
        line_cws = jieba_cws(line)
        sys.stdout.write(str(line_cws.strip()))
        sys.stdout.write('\n')
