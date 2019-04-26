import os
import argparse
import json

parser = argparse.ArgumentParser(description='json')
parser.add_argument('--input', type=str, default=None,
                    help='path to input')
parser.add_argument('--output', type=str, default=None,
                    help='path to input')
args = parser.parse_args()


def one_line_parse(string):
    msg = json.loads(string, strict=False)
    return msg["ja"].strip(), msg["zh"].strip()


if __name__ == '__main__':
    with open(args.input, "r") as f:
        with open(os.path.join(args.output, "ja-zh_ja.txt"), "w") as g, \
                open(os.path.join(args.output, "ja-zh_zh.txt"), "w") as h:
            for line in f.readlines():
                line = line.strip()
                try:
                    ja, zh = one_line_parse(line)
                    g.write(ja + "\n")
                    h.write(zh + "\n")
                except Exception as e:
                    print(e)
                    continue
