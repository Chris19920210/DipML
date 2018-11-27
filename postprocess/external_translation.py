import urllib.request
import argparse
import os
import json
from urllib.request import quote, unquote

parser = argparse.ArgumentParser(description='get')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to data base dir')
parser.add_argument('--input', type=str, required=True,
                    help='input path for ensemble file')
parser.add_argument('--output', type=str, required=True,
                    help='output path for ensemble result')
args = parser.parse_args()

if __name__ == '__main__':
    prefix = "http://39.104.68.130:1717/NiuTransServer/translation?from=en&to=zh&src_text="
    with open(os.path.join(args.base_dir, args.output), 'w') as g:
        with open(os.path.join(args.base_dir, args.input), 'r') as f:
            for each in f.readlines():
                req = urllib.request.Request(prefix+quote(each.strip()))
                response = urllib.request.urlopen(req)
                ret = json.loads(response.read())
                g.write(ret["tgt_text"].strip())
                g.write("\n")
