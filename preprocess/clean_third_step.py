import sys
import html


def removers(line):
    line = line.strip()
    zhline = html.unescape(line.split("\t")[0])
    enline = html.unescape(line.split("\t")[1])
    if len(zhline.split(" ")) < 2 or len(enline.split(" ")) < 2:
        return -1
    return zhline + "\t" + enline


if __name__ == '__main__':
    for line in sys.stdin:
        line_cws = removers(line)
        if line_cws != -1:
            sys.stdout.write(line_cws)
            sys.stdout.write('\n')
