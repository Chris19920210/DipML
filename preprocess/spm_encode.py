from SpmTextEncoder import SpmTextEncoder
import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bpe')
    parser.add_argument('--model', type=str, default=None,
                        help='spm model')
    args = parser.parse_args()

    spm = SpmTextEncoder(filename=args.model)

    for line in sys.stdin:
        line_cws = spm.encode_as_pieces(line.strip())
        sys.stdout.write(" ".join(line_cws))
        sys.stdout.write('\n')
