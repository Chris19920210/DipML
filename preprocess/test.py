from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
parser = argparse.ArgumentParser(description='remover')
parser.add_argument('--base-dir', type=str, default=None,
                    help='path to base-dir')
args = parser.parse_args()
print(args.base_dir)