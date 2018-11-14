import os
import argparse
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu

parser = argparse.ArgumentParser(description='parallel sentence presentation')
parser.add_argument('--base-dir', type=str, default="./",
                    help='path to data base dir')
parser.add_argument('--src', type=str, default='src',
                    help='source')
parser.add_argument('--ref', type=str,  default='ref.seg',
                    help='reference')
parser.add_argument('--hyp', type=str,  default='hyp.seg',
                    help='translated')
parser.add_argument('--output', type=str, default='sentence.bleu',
                    help='translated')
args = parser.parse_args()

if __name__ == '__main__':
    src_reader = open(os.path.join(args.base_dir, args.src), 'r')
    ref_reader = open(os.path.join(args.base_dir, args.ref), 'r')
    hyp_reader = open(os.path.join(args.base_dir, args.hyp), 'r')
    result = []
    smoothie = SmoothingFunction().method4
    for src, ref, hyp in zip(src_reader.readlines(), ref_reader.readlines(), hyp_reader.readlines()):
        score = sentence_bleu([ref.strip().split(" ")], hyp.strip().split(" "), smoothing_function=smoothie)
        result.append((src.strip(), ref.strip(), hyp.strip(), str(score)))
    result.sort(key=lambda x: x[3])

    with open(os.path.join(args.base_dir, args.output), 'w') as g:
        for index, line in enumerate(result):
            g.write(str(index))
            g.write('\t')
            g.write('\t'.join(line))
            g.write('\n')

