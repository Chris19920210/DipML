import re
import sys


pattern_1 = re.compile("^(\" ){0,}(.*)$")
pattern_2 = re.compile("\" \"")


def quota_remover(zhline, enline):
    global pattern_1, pattern_2

    zhline = re.sub(pattern_1, r'\2', zhline)
    zhline = re.sub(pattern_1, r'\2', zhline[::-1])
    zhline = zhline[::-1]

    enline = re.sub(pattern_1, r'\2', enline)
    enline = re.sub(pattern_1, r'\2', enline[::-1])
    enline = enline[::-1]

    enline = re.sub(pattern_2, "\"", enline)

    return zhline, enline


pattern_3 = re.compile("[\-]{2,}|[\-\s]{2,}")


def hyphen_remover(zhline, enline):
    global pattern_3
    return re.sub(pattern_3, "--", zhline), re.sub(pattern_3, "--", enline)


def removers(line):
    line = line.strip()
    zhline = line.split("\t")[0]
    enline = line.split("\t")[1]
    zhline, enline = quota_remover(zhline, enline)
    zhline, enline = hyphen_remover(zhline, enline)
    return zhline + '\t' + enline


if __name__ == '__main__':
    for line in sys.stdin:
        line_cws = removers(line)
        sys.stdout.write(line_cws)
        sys.stdout.write('\n')


