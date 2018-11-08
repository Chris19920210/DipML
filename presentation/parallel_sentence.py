import os
import argparse
parser = argparse.ArgumentParser(description='parallel sentence presentation')
parser.add_argument('--base-dir', type=str, default="./",
                    help='path to data base dir')
parser.add_argument('--src', type=str, default='valid.en-zh.en',
                    help='source')
parser.add_argument('--ref', type=str,  default='valid.en-zh.zh',
                    help='reference')
parser.add_argument('--hyp', type=str,  default='mt_trans.plain',
                    help='translated')
parser.add_argument('--output', type=str, default='parallel_result.txt',
                    help='translated')
args = parser.parse_args()

if __name__ == '__main__':
    src_reader = open(os.path.join(args.base_dir, args.src), 'r')
    ref_reader = open(os.path.join(args.base_dir, args.ref), 'r')
    hyp_reader = open(os.path.join(args.base_dir, args.hyp), 'r')
    with open(os.path.join(args.base_dir, args.output), 'w') as g:
        for src, ref, hyp in zip(src_reader.readlines(), ref_reader.readlines(), hyp_reader.readlines()):
            g.write(ref.strip())
            g.write('\t')
            g.write(src.strip())
            g.write('\t')
            g.write(hyp.strip())
            g.write('\n')
